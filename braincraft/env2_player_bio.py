# Braincraft challenge — Bio Player for Environment 2
# Copyright (C) 2026 Guanchun Li
# Released under the GNU General Public License 3

"""
Bio Player for Environment 2.

A pointwise-activation Echo State Network controller: every hidden
activation f(x)[i] depends only on x[i], and all cross-neuron logic is
carried by the connectivity matrices. The update is

    X(t+1) = f(Win @ I(t) + W @ X(t))
    O(t+1) = Wout @ g(X(t+1))

with identity readout g(x) = x.

Env2 adds a per-ray colour channel, so the input vector is

    I(t) = [prox[0..63](t), colour[0..63](t), hit(t), energy(t), 1]
                                                       (2p + 3 = 131 cols)

The hidden pool packs one functional slot per neuron:

    0..4    reflex features (hit, proximity, safety)
    5..8    dtheta, integrated heading, position accumulators
    9..13   initial-heading correction latch
    14..19  energy-based reward circuit and shortcut actuators
    20..31  trig helpers, corridor predicates, shortcut trigger
    32..49  phase gates, quadrant ANDs, blue-evidence front-block
    50..    per-ray blue-bump detectors

Constants use snake_case; local hidden-state aliases use UPPER_SNAKE.
"""

import numpy as np

if not hasattr(np, "atan2"):
    np.atan2 = np.arctan2

from bot import Bot
from environment_2 import Environment


# ── Shortcut circuit parameters ───────────────────────────────────────
shortcut_turn  = -2.0      # saturated steering magnitude inside the turn
near_c_thr     = 0.05      # half-width of the corridor bump detectors
drift_offset   = 0.175     # pos_x offset where the shortcut is armed
turn_steps     = 18        # length of the hard-turn phase
approach_steps = 50        # length of the straight approach phase
sc_total       = turn_steps + approach_steps

# ── Bio-specific constants ────────────────────────────────────────────
color_evidence_thr = 2.0             # integrated blue evidence needed to latch front sign
front_gain_mag     = np.radians(20.0)
gate_c             = 1.0
k_sharp            = 50.0            # logistic-like gain for AND/OR/latch gates
step_a             = np.radians(5.0) # actuator clip (±5°)
seed_window_k      = 6               # initial-heading correction window length


# ── Neuron index layout ───────────────────────────────────────────────
def _bio_indices(n_rays):
    """Sequential neuron-slot map for the bio controller."""
    idx = {
        "hit_feat":       0,
        "prox_left":      1,
        "prox_right":     2,
        "safe_left":      3,
        "safe_right":     4,
        "dtheta":         5,
        "dir_accum":      6,
        "pos_x":          7,
        "pos_y":          8,
        "head_corr":      9,
        "seeded_flag":   10,
        "step_counter":  11,
        "seed_pos":      12,
        "seed_neg":      13,
        "energy_ramp":   14,
        "reward_pulse":  15,
        "reward_latch":  16,
        "sc_countdown":  17,
        "shortcut_steer":18,
        "init_impulse":  19,
        "sin_n":         20,
        "cos_n":         21,
        "sin_pos":       22,
        "sin_neg":       23,
        "y_pos":         24,
        "y_neg":         25,
        "near_e":        26,
        "near_w":        27,
        "near_cr_e":     28,
        "near_cr_w":     29,
        "near_cr":       30,
        "trig_sc":       31,
        "on_countdown":  32,
        "is_turn":       33,
        "is_app":        34,
        "sy_pp":         35,
        "sy_pn":         36,
        "sy_np":         37,
        "sy_nn":         38,
        "front_block_pos": 39,
        "front_block_neg": 40,
        "l_ev":          41,
        "r_ev":          42,
        "dleft":         43,
        "dright":        44,
        "evidence":      45,
        "trig_pos":      46,
        "trig_neg":      47,
        "fs_pos":        48,
        "fs_neg":        49,
    }
    idx["xi_blue_start"] = 50
    idx["xi_blue_stop"]  = idx["xi_blue_start"] + n_rays
    idx["half"]          = n_rays // 2
    idx["bio_end"]       = idx["xi_blue_stop"]
    return idx


# ── Pointwise activation dispatch ─────────────────────────────────────
def make_activation(a, idx):
    """Build the per-neuron pointwise activation.

    Each out[i] depends only on x[i]. The activation used for neuron i is
    picked once here from precomputed index arrays:

        identity  : linear slots (integrators, accumulators, sums)
        sin       : trig helpers (sin_n, cos_n)
        relu      : energy_ramp, sc_countdown
        bump      : max(0, 1 - 4 z^2) — corridor detectors and blue rays
        clip_a    : dtheta, clipped to ±step_a
        default   : max(0, tanh(z)) — sparse threshold / latch gates
    """
    id_list = [
        idx["dir_accum"], idx["pos_x"], idx["pos_y"], idx["head_corr"],
        idx["shortcut_steer"], idx["init_impulse"],
        idx["evidence"], idx["l_ev"], idx["r_ev"],
        idx["step_counter"],
    ]
    sin_list    = [idx["sin_n"], idx["cos_n"]]
    relu_list   = [idx["energy_ramp"], idx["sc_countdown"]]
    bump_list   = [idx["near_e"], idx["near_w"]]
    bump_list.extend(range(idx["xi_blue_start"], idx["xi_blue_stop"]))

    id_arr     = np.array(sorted(set(id_list)),     dtype=int)
    sin_arr    = np.array(sorted(set(sin_list)),    dtype=int)
    relu_arr   = np.array(sorted(set(relu_list)),   dtype=int)
    bump_arr   = np.array(sorted(set(bump_list)),   dtype=int)

    def f(x):
        out = np.maximum(0.0, np.tanh(x))  # default relu_tanh
        if id_arr.size:
            out[id_arr, 0] = x[id_arr, 0]
        if sin_arr.size:
            out[sin_arr, 0] = np.sin(x[sin_arr, 0])
        if relu_arr.size:
            out[relu_arr, 0] = np.maximum(0.0, x[relu_arr, 0])
        if bump_arr.size:
            out[bump_arr, 0] = np.maximum(0.0, 1.0 - 4.0 * x[bump_arr, 0] ** 2)
        out[idx["dtheta"], 0] = float(np.clip(x[idx["dtheta"], 0], -a, a))
        return out

    return f


# ── Player builder ────────────────────────────────────────────────────
def bio_player():
    """Build the bio controller for env2. Yields a single frozen model."""

    bot = Bot()
    n = 1000
    p = bot.camera.resolution          # 64
    warmup = 0
    leak = 1.0
    g = lambda x: x                    # identity readout

    n_inputs = 2 * p + 3
    Win  = np.zeros((n, n_inputs))
    W    = np.zeros((n, n))
    Wout = np.zeros((1, n))

    hit_idx    = 2 * p                 # 128
    energy_idx = 2 * p + 1             # 129
    bias_idx   = 2 * p + 2             # 130

    speed  = bot.speed                 # 0.01
    n_rays = p                         # 64
    idx = _bio_indices(n_rays)
    a = step_a

    # Local aliases — keep wiring lines short and readable.
    HIT_FEAT       = idx["hit_feat"]
    PROX_LEFT      = idx["prox_left"]
    PROX_RIGHT     = idx["prox_right"]
    SAFE_LEFT      = idx["safe_left"]
    SAFE_RIGHT     = idx["safe_right"]
    DTHETA         = idx["dtheta"]
    DIR_ACCUM      = idx["dir_accum"]
    POS_X          = idx["pos_x"]
    POS_Y          = idx["pos_y"]
    HEAD_CORR      = idx["head_corr"]
    SEEDED_FLAG    = idx["seeded_flag"]
    STEP_COUNTER   = idx["step_counter"]
    ENERGY_RAMP    = idx["energy_ramp"]
    REWARD_PULSE   = idx["reward_pulse"]
    REWARD_LATCH   = idx["reward_latch"]
    SC_COUNTDOWN   = idx["sc_countdown"]
    SHORTCUT_STEER = idx["shortcut_steer"]
    INIT_IMPULSE   = idx["init_impulse"]
    SIN_N          = idx["sin_n"]
    COS_N          = idx["cos_n"]
    SIN_POS        = idx["sin_pos"]
    SIN_NEG        = idx["sin_neg"]
    Y_POS          = idx["y_pos"]
    Y_NEG          = idx["y_neg"]
    NEAR_E         = idx["near_e"]
    NEAR_W         = idx["near_w"]
    NEAR_CR        = idx["near_cr"]
    TSC            = idx["trig_sc"]
    ONC            = idx["on_countdown"]
    IST            = idx["is_turn"]
    ISA            = idx["is_app"]
    SEEDP          = idx["seed_pos"]
    SEEDN          = idx["seed_neg"]
    SY_PP          = idx["sy_pp"]
    SY_PN          = idx["sy_pn"]
    SY_NP          = idx["sy_np"]
    SY_NN          = idx["sy_nn"]
    FBP            = idx["front_block_pos"]
    FBN            = idx["front_block_neg"]
    L_EV           = idx["l_ev"]
    R_EV           = idx["r_ev"]
    DLEFT          = idx["dleft"]
    DRIGHT         = idx["dright"]
    EV             = idx["evidence"]
    TP             = idx["trig_pos"]
    TN             = idx["trig_neg"]
    FS_P           = idx["fs_pos"]
    FS_N           = idx["fs_neg"]

    # Ray indices sampled by the reflex/front channels.
    L_idx          = 20
    R_idx          = 63 - L_idx         # 43
    left_side_idx  = 11
    right_side_idx = 63 - left_side_idx # 52
    C1_idx, C2_idx = 31, 32              # two centre-front proximity taps
    front_thr      = 1.4

    TANH1 = np.tanh(1.0)

    hit_turn          = np.radians(-10.0) / TANH1
    heading_gain      = np.radians(-40.0)
    safety_gain_left  = np.radians(-20.0)
    safety_gain_right = -safety_gain_left
    safety_target     = 0.75

    # ── Reflex features and steering readout ──────────────────────────
    Win[HIT_FEAT,    hit_idx]        = 1.0
    Win[PROX_LEFT,   L_idx]          = 1.0
    Win[PROX_RIGHT,  R_idx]          = 1.0
    Win[SAFE_LEFT,   left_side_idx]  = -1.0
    Win[SAFE_RIGHT,  right_side_idx] = -1.0
    Win[SAFE_LEFT,   bias_idx]       = safety_target
    Win[SAFE_RIGHT,  bias_idx]       = safety_target

    # Silence reflex features during the shortcut approach so the output
    # reduces to the front-block + shortcut-steer + initial-impulse terms.
    for reflex_idx in (HIT_FEAT, PROX_LEFT, PROX_RIGHT, SAFE_LEFT, SAFE_RIGHT):
        W[reflex_idx, ISA] = -k_sharp

    Wout[0, HIT_FEAT]       = hit_turn
    Wout[0, PROX_LEFT]      = heading_gain
    Wout[0, PROX_RIGHT]     = -heading_gain
    Wout[0, SAFE_LEFT]      = safety_gain_left
    Wout[0, SAFE_RIGHT]     = safety_gain_right
    Wout[0, SHORTCUT_STEER] = 1.0
    Wout[0, INIT_IMPULSE]   = 1.0

    # ── Heading, trig, and position accumulators ──────────────────────
    # dir_accum integrates every applied dtheta. phi(t) is reconstructed
    # inside the trig neurons from dir_accum + head_corr + dtheta (their
    # three incoming weights of 1). W[DTHETA, :] is filled at the end
    # so that z_dtheta(t+1) = Wout @ X(t) = O(t).
    W[DIR_ACCUM, DIR_ACCUM] = 1.0
    W[DIR_ACCUM, DTHETA]    = 1.0

    # Env frame: dx = -speed * sin(phi), dy = +speed * cos(phi).
    W[POS_X, POS_X] = 1.0
    W[POS_X, SIN_N] = -speed
    W[POS_Y, POS_Y] = 1.0
    W[POS_Y, COS_N] = speed

    # ── Initial-heading correction latch ──────────────────────────────
    # step_counter is a plain identity counter (= t). seeded_flag is a
    # sharp threshold against it: 0 for t = 0..seed_window_k - 1, then
    # saturates to 1. seed_pos / seed_neg read seeded_flag(t), so they
    # fire for exactly seed_window_k consecutive steps (t = 1..K).
    # head_corr integrates their difference across that window, driving
    # the residual left/right depth asymmetry toward zero in closed loop.
    cal_gain = 1.0 / 0.173

    W[STEP_COUNTER, STEP_COUNTER] = 1.0
    Win[STEP_COUNTER, bias_idx]   = 1.0

    W[SEEDED_FLAG, STEP_COUNTER]  = k_sharp
    Win[SEEDED_FLAG, bias_idx]    = -k_sharp * (float(seed_window_k) - 1.5)

    # SEED_POS/NEG: signed (R - L) depth asymmetry, gated off once
    # seeded_flag saturates (-1000 gate drives the pre-activation well
    # below zero so relu_tanh outputs zero).
    Win[SEEDP, L_idx]             = -cal_gain
    Win[SEEDP, R_idx]             =  cal_gain
    W[SEEDP, SEEDED_FLAG]         = -1.0e3
    Win[SEEDN, L_idx]             =  cal_gain
    Win[SEEDN, R_idx]             = -cal_gain
    W[SEEDN, SEEDED_FLAG]         = -1.0e3

    W[HEAD_CORR, HEAD_CORR] = 1.0
    W[HEAD_CORR, SEEDP]     =  1.0
    W[HEAD_CORR, SEEDN]     = -1.0

    # init_impulse negates the active seed so its net steering
    # contribution cancels head_corr's integrated response during the
    # correction window.
    W[INIT_IMPULSE, SEEDP] = -1.0
    W[INIT_IMPULSE, SEEDN] =  1.0

    # ── Reward circuit ────────────────────────────────────────────────
    # energy_ramp holds energy(t-1). reward_pulse detects a rising edge
    # in energy(t) - energy(t-1) once seeded_flag arms it.
    pulse_gain = 500.0
    pulse_thr  = 0.2
    arm_gate   = 1000.0
    latch_gain = 10.0

    Win[ENERGY_RAMP, energy_idx]  = 1.0
    Win[REWARD_PULSE, energy_idx] = pulse_gain
    W[REWARD_PULSE, ENERGY_RAMP]  = -pulse_gain
    W[REWARD_PULSE, SEEDED_FLAG]  = arm_gate
    Win[REWARD_PULSE, bias_idx]   = -(arm_gate + pulse_thr)

    # reward_latch: hysteretic OR — fires on any pulse and holds.
    W[REWARD_LATCH, REWARD_PULSE] = latch_gain
    W[REWARD_LATCH, REWARD_LATCH] = latch_gain

    # ── Shortcut countdown and phase gates ────────────────────────────
    # sc_countdown is a relu integrator that decrements by 1 each step
    # and is reloaded to sc_total when trig_sc fires.
    W[SC_COUNTDOWN, SC_COUNTDOWN] = 1.0
    Win[SC_COUNTDOWN, bias_idx]   = -1.0
    W[SC_COUNTDOWN, TSC]          = float(sc_total) + 1.0

    # on_countdown: sc_countdown(t) > 0.5.
    W[ONC, SC_COUNTDOWN]          = k_sharp
    Win[ONC, bias_idx]            = -0.5 * k_sharp

    # is_turn: sc_countdown(t) > approach_steps + 0.5.
    W[IST, SC_COUNTDOWN]          = k_sharp
    Win[IST, bias_idx]            = -k_sharp * (float(approach_steps) + 0.5)

    # is_app: on_countdown AND NOT is_turn.
    W[ISA, ONC]                   =  k_sharp
    W[ISA, IST]                   = -k_sharp
    Win[ISA, bias_idx]            = -0.5 * k_sharp

    # Shortcut steering: turn_toward = sign(sin(phi)) * sign(y),
    # enabled only while is_turn is high.
    W[SHORTCUT_STEER, SY_PP] =  abs(shortcut_turn)  # sin+, y+
    W[SHORTCUT_STEER, SY_NN] =  abs(shortcut_turn)  # sin-, y-
    W[SHORTCUT_STEER, SY_PN] = -abs(shortcut_turn)  # sin+, y-
    W[SHORTCUT_STEER, SY_NP] = -abs(shortcut_turn)  # sin-, y+

    # ── Trig neurons (sin-only activation) ────────────────────────────
    # sin_n = sin(phi), cos_n = sin(phi + pi/2) = cos(phi).
    for trig_i in (SIN_N, COS_N):
        W[trig_i, DIR_ACCUM] = 1.0
        W[trig_i, HEAD_CORR] = 1.0
        W[trig_i, DTHETA]    = 1.0
    Win[COS_N, bias_idx] = np.pi / 2

    # Sharp sign detectors on sin(phi) and pos_y, used by the quadrant ANDs.
    W[SIN_POS, SIN_N] =  k_sharp
    W[SIN_NEG, SIN_N] = -k_sharp
    W[Y_POS,   POS_Y] =  k_sharp
    W[Y_NEG,   POS_Y] = -k_sharp

    # ── Corridor tests ────────────────────────────────────────────────
    # Bump half-width is 0.5 in pre-activation space, so scale pos_x by
    # 1/(2*near_c_thr) to get a bump width of ±near_c_thr in world units.
    bump_scale = 1.0 / (2.0 * near_c_thr)

    # near_e fires when pos_x is near -drift_offset (west of centre,
    # heading east); near_w is the mirror.
    W[NEAR_E, POS_X]   = bump_scale
    Win[NEAR_E, bias_idx] =  drift_offset * bump_scale
    W[NEAR_W, POS_X]   = bump_scale
    Win[NEAR_W, bias_idx] = -drift_offset * bump_scale

    # Heading-gated corridor-entry detectors. The internal heading is
    # represented relative to north, so sin_n = sin(phi_internal): east is
    # near -1 and west is near +1. The AND threshold at +/-0.5 accepts
    # headings within roughly +/-60 degrees of horizontal, while rejecting
    # perpendicular north/south crossings of pos_x = +/-drift_offset.
    ncr_gain = 2.5
    NEAR_CR_E = idx["near_cr_e"]
    NEAR_CR_W = idx["near_cr_w"]
    W[NEAR_CR_E, NEAR_E]      =  ncr_gain * k_sharp
    W[NEAR_CR_E, SIN_N]       = -ncr_gain * k_sharp    # sin_n ~= -1 => east
    Win[NEAR_CR_E, bias_idx]  = -ncr_gain * k_sharp * 1.5
    W[NEAR_CR_W, NEAR_W]      =  ncr_gain * k_sharp
    W[NEAR_CR_W, SIN_N]       =  ncr_gain * k_sharp    # sin_n ~= +1 => west
    Win[NEAR_CR_W, bias_idx]  = -ncr_gain * k_sharp * 1.5

    # near_cr = near_cr_e OR near_cr_w. The 2.5x gain keeps the OR-gate
    # transition sharp across BLAS-level roundoff in the inputs.
    near_cr_gain = 2.5
    W[NEAR_CR, NEAR_CR_E] = near_cr_gain * k_sharp
    W[NEAR_CR, NEAR_CR_W] = near_cr_gain * k_sharp
    Win[NEAR_CR, bias_idx] = -0.5 * near_cr_gain * k_sharp

    # ── Shortcut trigger (2-way AND with refractory) ──────────────────
    # trig_sc fires when reward has been seen AND the heading-gated
    # corridor-entry neuron fires. near_cr_e/near_cr_w already encode
    # "position at corridor entry + heading toward the clear corridor
    # axis", which is a sufficient precondition on its own.
    W[TSC, REWARD_LATCH] = k_sharp
    W[TSC, NEAR_CR]      = k_sharp
    Win[TSC, bias_idx]   = -k_sharp * 1.5
    W[TSC, TSC]          = -k_sharp * 10     # self-refractory
    W[TSC, SC_COUNTDOWN] = -k_sharp           # blocked while countdown runs

    # ── Quadrant ANDs for shortcut steering direction ─────────────────
    # Each sy_* = AND(sin-sign, y-sign, is_turn), implemented as a
    # 3-input threshold gate.
    W[SY_PP, SIN_POS] = k_sharp
    W[SY_PP, Y_POS]   = k_sharp
    W[SY_PP, IST]     = k_sharp
    Win[SY_PP, bias_idx] = -2.5 * k_sharp

    W[SY_PN, SIN_POS] = k_sharp
    W[SY_PN, Y_NEG]   = k_sharp
    W[SY_PN, IST]     = k_sharp
    Win[SY_PN, bias_idx] = -2.5 * k_sharp

    W[SY_NP, SIN_NEG] = k_sharp
    W[SY_NP, Y_POS]   = k_sharp
    W[SY_NP, IST]     = k_sharp
    Win[SY_NP, bias_idx] = -2.5 * k_sharp

    W[SY_NN, SIN_NEG] = k_sharp
    W[SY_NN, Y_NEG]   = k_sharp
    W[SY_NN, IST]     = k_sharp
    Win[SY_NN, bias_idx] = -2.5 * k_sharp

    # ── Blue-evidence accumulator and front-block sign latch ──────────
    # Each xi_blue[r] is a unit bump centred at colour value 4 (= blue).
    for r in range(n_rays):
        color_input_col = p + r
        Win[idx["xi_blue_start"] + r, color_input_col] = 1.0
        Win[idx["xi_blue_start"] + r, bias_idx]        = -4.0

    half = idx["half"]
    for r in range(half):
        W[L_EV, idx["xi_blue_start"] + r] = 1.0
    for r in range(half, n_rays):
        W[R_EV, idx["xi_blue_start"] + r] = 1.0

    # dleft / dright are one-sided dominance pulses, gated off once either
    # front-sign latch is set (so the signed integrator stops updating).
    W[DLEFT, L_EV]       =  k_sharp
    W[DLEFT, R_EV]       = -k_sharp
    W[DLEFT, FS_P]       = -k_sharp * 10
    W[DLEFT, FS_N]       = -k_sharp * 10
    Win[DLEFT, bias_idx] = -0.2 * k_sharp

    W[DRIGHT, L_EV]       = -k_sharp
    W[DRIGHT, R_EV]       =  k_sharp
    W[DRIGHT, FS_P]       = -k_sharp * 10
    W[DRIGHT, FS_N]       = -k_sharp * 10
    Win[DRIGHT, bias_idx] = -0.2 * k_sharp

    # evidence = signed integrator of (dright - dleft).
    W[EV, EV]     =  1.0
    W[EV, DRIGHT] =  1.0
    W[EV, DLEFT]  = -1.0

    # trig_pos / trig_neg: sharp thresholds at ±color_evidence_thr.
    # DLEFT/DRIGHT saturate to ~1, so shifting by 0.5 makes the trigger
    # fire exactly when evidence crosses color_evidence_thr.
    W[TP, EV] =  k_sharp
    Win[TP, bias_idx] = -k_sharp * (color_evidence_thr - 0.5)
    W[TN, EV] = -k_sharp
    Win[TN, bias_idx] = -k_sharp * (color_evidence_thr - 0.5)

    # Front-sign latches.
    W[FS_P, FS_P] = k_sharp
    W[FS_P, TP]   = k_sharp
    W[FS_N, FS_N] = k_sharp
    W[FS_N, TN]   = k_sharp

    # Gated front-block channels: trigger when the two centre proximity
    # taps exceed front_thr, with the active sign channel biased open
    # and the opposite channel biased shut by gate_c.
    Win[FBP, C1_idx]   = 1.0
    Win[FBP, C2_idx]   = 1.0
    Win[FBP, bias_idx] = -(front_thr + gate_c)
    W[FBP, FS_P]       =  gate_c
    W[FBP, FS_N]       = -gate_c

    Win[FBN, C1_idx]   = 1.0
    Win[FBN, C2_idx]   = 1.0
    Win[FBN, bias_idx] = -(front_thr + gate_c)
    W[FBN, FS_P]       = -gate_c
    W[FBN, FS_N]       =  gate_c

    Wout[0, FBP] =  front_gain_mag
    Wout[0, FBN] = -front_gain_mag

    # ── dtheta readout tie-back ───────────────────────────────────────
    # z_dtheta(t+1) = Wout @ X(t) = O(t), implemented by mirroring the
    # output row into W[DTHETA, :]. After clipping, dtheta(t+1) is the
    # previous step's steering command bounded to ±step_a.
    for j in range(n):
        if Wout[0, j] != 0.0:
            W[DTHETA, j] = Wout[0, j]

    f = make_activation(a, idx)
    model = Win, W, Wout, warmup, leak, f, g
    yield model


if __name__ == "__main__":
    import time
    from challenge_2 import evaluate, train

    seed = 12345
    np.random.seed(seed)
    print("Training bio player for env2...")
    model = train(bio_player, timeout=100)

    W_in, W, W_out, warmup, leak, f, g = model

    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score (distance): {score:.2f} +/- {std:.2f}")
