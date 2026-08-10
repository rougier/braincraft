# Braincraft challenge — Bio Player for Environment 1
# Copyright (C) 2026 Guanchun Li
# Released under the GNU General Public License 3

"""
Bio Player for Environment 1.

A pointwise-activation Echo State Network controller. Every hidden
activation depends only on its own preactivation; all cross-neuron logic
is carried by the connectivity matrices:

    X(t+1) = f(Win @ I(t) + W @ X(t))
    O(t+1) = Wout @ g(X(t+1))        (g = identity)

Input (67 cols): I(t) = [prox[0..63](t), hit(t), energy(t), 1].

Hidden slots (40 active, rest unused):

    0..4    reflex features (hit, proximity, safety)
    5..8    dtheta, integrated heading, position accumulators
    9..13   initial-heading correction latch
    14..19  reward circuit, shortcut countdown/steer, initial-impulse
    20..31  trig helpers, corridor predicates, shortcut trigger
    32..39  phase gates, quadrant ANDs, unsigned front-block

The controller combines four behaviours: a reflex wall-follower, a
pose-gated corridor shortcut, a rising-edge energy-reward detector, and
an initial-heading correction that erases the ±5° start-direction
perturbation.
"""

import numpy as np

if not hasattr(np, "atan2"):
    np.atan2 = np.arctan2

from bot import Bot
from environment_1 import Environment


# ── Shortcut circuit parameters ───────────────────────────────────────
shortcut_turn  = -2.0      # saturated steering magnitude inside the turn
near_c_thr     = 0.05      # half-width of the corridor bump detectors
drift_offset   = 0.175     # pos_x offset where the shortcut is armed
turn_steps     = 18        # length of the hard-turn phase
approach_steps = 50        # length of the straight approach phase
sc_total       = turn_steps + approach_steps

# ── Other constants ───────────────────────────────────────────────────
front_gain_mag = np.radians(20.0)
k_sharp        = 50.0              # gain for AND/OR/latch gates
step_a         = np.radians(5.0)   # actuator clip (±5°)
seed_window_k  = 6                 # initial-heading correction window length


# ── Neuron index layout ───────────────────────────────────────────────
def _bio_indices():
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
        "front_block":   39,
    }
    idx["bio_end"] = 40
    return idx


# ── Pointwise activation dispatch ─────────────────────────────────────
def make_activation(a, idx):
    """Per-neuron pointwise activation; each slot uses a fixed scalar fn.

        identity  : linear slots (integrators, accumulators, counters)
        sin       : trig helpers (sin_n, cos_n)
        relu      : energy_ramp, sc_countdown
        bump      : max(0, 1 - 4 z^2) — corridor detectors
        clip_a    : dtheta, clipped to ±step_a
        default   : max(0, tanh(z)) — threshold / latch / AND-OR gates
    """
    id_arr   = np.array(sorted({idx["dir_accum"], idx["pos_x"], idx["pos_y"],
                                idx["head_corr"], idx["shortcut_steer"],
                                idx["init_impulse"], idx["step_counter"]}),
                        dtype=int)
    sin_arr  = np.array(sorted({idx["sin_n"], idx["cos_n"]}), dtype=int)
    relu_arr = np.array(sorted({idx["energy_ramp"], idx["sc_countdown"]}),
                        dtype=int)
    bump_arr = np.array(sorted({idx["near_e"], idx["near_w"]}), dtype=int)

    def f(x):
        out = np.maximum(0.0, np.tanh(x))  # default relu_tanh
        out[id_arr, 0]   = x[id_arr, 0]
        out[sin_arr, 0]  = np.sin(x[sin_arr, 0])
        out[relu_arr, 0] = np.maximum(0.0, x[relu_arr, 0])
        out[bump_arr, 0] = np.maximum(0.0, 1.0 - 4.0 * x[bump_arr, 0] ** 2)
        out[idx["dtheta"], 0] = float(np.clip(x[idx["dtheta"], 0], -a, a))
        return out

    return f


# ── Player builder ────────────────────────────────────────────────────
def bio_player():
    """Build the env1 bio controller and yield a single frozen model."""

    bot = Bot()
    n = 1000
    p = bot.camera.resolution          # 64
    warmup = 0
    leak = 1.0
    g = lambda x: x                    # identity readout

    n_inputs = p + 3                   # 67: prox(64) + hit + energy + bias
    Win  = np.zeros((n, n_inputs))
    W    = np.zeros((n, n))
    Wout = np.zeros((1, n))

    hit_idx    = p                     # 64
    energy_idx = p + 1                 # 65
    bias_idx   = p + 2                 # 66

    speed = bot.speed                  # 0.01
    idx   = _bio_indices()
    a     = step_a

    # Short aliases for the wiring block.
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
    SEEDP          = idx["seed_pos"]
    SEEDN          = idx["seed_neg"]
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
    NEAR_CR_E      = idx["near_cr_e"]
    NEAR_CR_W      = idx["near_cr_w"]
    NEAR_CR        = idx["near_cr"]
    TSC            = idx["trig_sc"]
    ONC            = idx["on_countdown"]
    IST            = idx["is_turn"]
    ISA            = idx["is_app"]
    SY_PP          = idx["sy_pp"]
    SY_PN          = idx["sy_pn"]
    SY_NP          = idx["sy_np"]
    SY_NN          = idx["sy_nn"]
    FB             = idx["front_block"]

    # Ray taps.
    L_idx, R_idx                  = 20, 43     # reflex proximity taps
    left_side_idx, right_side_idx = 11, 52     # safety taps
    C1_idx, C2_idx                = 31, 32     # centre-front proximity taps
    front_thr                     = 1.4

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

    # Silence reflexes during shortcut approach — readout then reduces
    # to front_block + shortcut_steer + init_impulse.
    for r in (HIT_FEAT, PROX_LEFT, PROX_RIGHT, SAFE_LEFT, SAFE_RIGHT):
        W[r, ISA] = -k_sharp

    Wout[0, HIT_FEAT]       = hit_turn
    Wout[0, PROX_LEFT]      = heading_gain
    Wout[0, PROX_RIGHT]     = -heading_gain
    Wout[0, SAFE_LEFT]      = safety_gain_left
    Wout[0, SAFE_RIGHT]     = safety_gain_right
    Wout[0, SHORTCUT_STEER] = 1.0
    Wout[0, INIT_IMPULSE]   = 1.0

    # ── Heading, trig, and position accumulators ──────────────────────
    # phi(t) = dir_accum(t) + head_corr(t) + dtheta(t) is reconstructed
    # inside the trig neurons. W[DTHETA, :] is mirrored from Wout at the
    # end so that z_dtheta(t+1) = O(t).
    W[DIR_ACCUM, DIR_ACCUM] = 1.0
    W[DIR_ACCUM, DTHETA]    = 1.0

    # Env frame: dx = -speed * sin(phi), dy = +speed * cos(phi).
    W[POS_X, POS_X] = 1.0
    W[POS_X, SIN_N] = -speed
    W[POS_Y, POS_Y] = 1.0
    W[POS_Y, COS_N] = speed

    # ── Initial-heading correction latch ──────────────────────────────
    # step_counter counts steps; seeded_flag saturates once step_counter
    # crosses seed_window_k. seed_pos/seed_neg read the signed (R-L)
    # depth asymmetry and are gated off after seed_window_k steps. Inside
    # that window head_corr integrates the residual, and init_impulse
    # steers against it so the net steering contribution cancels — the
    # closed loop drives the residual heading error toward zero.
    cal_gain = 1.0 / 0.173

    W[STEP_COUNTER, STEP_COUNTER] = 1.0
    Win[STEP_COUNTER, bias_idx]   = 1.0

    W[SEEDED_FLAG, STEP_COUNTER]  = k_sharp
    Win[SEEDED_FLAG, bias_idx]    = -k_sharp * (float(seed_window_k) - 1.5)

    # seed_pos/_neg: signed (R-L) depth, gated off once seeded_flag=1.
    Win[SEEDP, L_idx]     = -cal_gain
    Win[SEEDP, R_idx]     =  cal_gain
    W[SEEDP, SEEDED_FLAG] = -1.0e3
    Win[SEEDN, L_idx]     =  cal_gain
    Win[SEEDN, R_idx]     = -cal_gain
    W[SEEDN, SEEDED_FLAG] = -1.0e3

    W[HEAD_CORR, HEAD_CORR] = 1.0
    W[HEAD_CORR, SEEDP]     =  1.0
    W[HEAD_CORR, SEEDN]     = -1.0

    # init_impulse negates the active seed so the bot doesn't physically
    # turn during the correction window.
    W[INIT_IMPULSE, SEEDP] = -1.0
    W[INIT_IMPULSE, SEEDN] =  1.0

    # ── Reward circuit ────────────────────────────────────────────────
    # reward_pulse = rising-edge on energy(t) - energy(t-1), gated by
    # seeded_flag so it fires only after the correction window closes.
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
    # sc_countdown decrements by 1 each step and is reloaded to sc_total
    # when trig_sc fires. is_turn / is_app split the countdown into a
    # hard-turn phase and a straight approach phase.
    W[SC_COUNTDOWN, SC_COUNTDOWN] = 1.0
    Win[SC_COUNTDOWN, bias_idx]   = -1.0
    W[SC_COUNTDOWN, TSC]          = float(sc_total) + 1.0

    W[ONC, SC_COUNTDOWN]          = k_sharp
    Win[ONC, bias_idx]            = -0.5 * k_sharp

    W[IST, SC_COUNTDOWN]          = k_sharp
    Win[IST, bias_idx]            = -k_sharp * (float(approach_steps) + 0.5)

    W[ISA, ONC]                   =  k_sharp
    W[ISA, IST]                   = -k_sharp
    Win[ISA, bias_idx]            = -0.5 * k_sharp

    # Shortcut steering: sign(sin(phi)) * sign(y), enabled only in turn.
    W[SHORTCUT_STEER, SY_PP] =  abs(shortcut_turn)
    W[SHORTCUT_STEER, SY_NN] =  abs(shortcut_turn)
    W[SHORTCUT_STEER, SY_PN] = -abs(shortcut_turn)
    W[SHORTCUT_STEER, SY_NP] = -abs(shortcut_turn)

    # ── Trig neurons (sin-only activation) ────────────────────────────
    # sin_n = sin(phi), cos_n = sin(phi + pi/2) = cos(phi).
    for trig_i in (SIN_N, COS_N):
        W[trig_i, DIR_ACCUM] = 1.0
        W[trig_i, HEAD_CORR] = 1.0
        W[trig_i, DTHETA]    = 1.0
    Win[COS_N, bias_idx] = np.pi / 2

    W[SIN_POS, SIN_N] =  k_sharp
    W[SIN_NEG, SIN_N] = -k_sharp
    W[Y_POS,   POS_Y] =  k_sharp
    W[Y_NEG,   POS_Y] = -k_sharp

    # ── Corridor tests ────────────────────────────────────────────────
    # Bump half-width is 0.5 in pre-activation space, so scale pos_x by
    # 1/(2*near_c_thr) for a ±near_c_thr bump in world units.
    bump_scale = 1.0 / (2.0 * near_c_thr)
    W[NEAR_E, POS_X]      = bump_scale
    Win[NEAR_E, bias_idx] =  drift_offset * bump_scale
    W[NEAR_W, POS_X]      = bump_scale
    Win[NEAR_W, bias_idx] = -drift_offset * bump_scale

    # Heading-gated corridor entries. phi is measured from north, and
    # sin_n = sin(phi), so sin_n ≈ -1 while heading east and ≈ +1 while
    # heading west. The ±0.5 margin accepts headings within ~±60° of
    # horizontal and rejects perpendicular crossings of
    # pos_x = ±drift_offset on later laps.
    ncr_gain = 2.5
    W[NEAR_CR_E, NEAR_E]     =  ncr_gain * k_sharp
    W[NEAR_CR_E, SIN_N]      = -ncr_gain * k_sharp       # east
    Win[NEAR_CR_E, bias_idx] = -ncr_gain * k_sharp * 1.5
    W[NEAR_CR_W, NEAR_W]     =  ncr_gain * k_sharp
    W[NEAR_CR_W, SIN_N]      =  ncr_gain * k_sharp       # west
    Win[NEAR_CR_W, bias_idx] = -ncr_gain * k_sharp * 1.5

    # near_cr = near_cr_e OR near_cr_w. The 2.5× gain keeps the OR-gate
    # transition sharp across BLAS-level roundoff.
    near_cr_gain = 2.5
    W[NEAR_CR, NEAR_CR_E]  = near_cr_gain * k_sharp
    W[NEAR_CR, NEAR_CR_W]  = near_cr_gain * k_sharp
    Win[NEAR_CR, bias_idx] = -0.5 * near_cr_gain * k_sharp

    # ── Shortcut trigger (2-way AND with refractory) ──────────────────
    # trig_sc fires when reward has been seen AND the heading-gated
    # corridor-entry neuron fires; two refractory terms block re-firing
    # during the countdown.
    W[TSC, REWARD_LATCH] = k_sharp
    W[TSC, NEAR_CR]      = k_sharp
    Win[TSC, bias_idx]   = -k_sharp * 1.5
    W[TSC, TSC]          = -k_sharp * 10    # self-refractory
    W[TSC, SC_COUNTDOWN] = -k_sharp          # blocked while countdown runs

    # ── Quadrant ANDs for shortcut steering direction ─────────────────
    # Each sy_* = AND(sin-sign, y-sign, is_turn).
    for sy, sx_sign, sy_sign in (
        (SY_PP, SIN_POS, Y_POS),
        (SY_PN, SIN_POS, Y_NEG),
        (SY_NP, SIN_NEG, Y_POS),
        (SY_NN, SIN_NEG, Y_NEG),
    ):
        W[sy, sx_sign]     = k_sharp
        W[sy, sy_sign]     = k_sharp
        W[sy, IST]         = k_sharp
        Win[sy, bias_idx]  = -2.5 * k_sharp

    # ── Front-block channel (unsigned) ────────────────────────────────
    # Fires when the two centre proximity taps exceed front_thr. A
    # positive reading turns the bot in a fixed direction (CCW) via
    # Wout — a simple bounce-off-the-wall escape.
    Win[FB, C1_idx]   = 1.0
    Win[FB, C2_idx]   = 1.0
    Win[FB, bias_idx] = -front_thr

    Wout[0, FB] = front_gain_mag

    # ── dtheta readout tie-back ───────────────────────────────────────
    # Mirror Wout into W[DTHETA, :] so dtheta(t+1) = clip(O(t), ±step_a).
    for j in range(n):
        if Wout[0, j] != 0.0:
            W[DTHETA, j] = Wout[0, j]

    f = make_activation(a, idx)
    model = Win, W, Wout, warmup, leak, f, g
    yield model


if __name__ == "__main__":
    import time
    from challenge_1 import evaluate, train

    seed = 12345
    np.random.seed(seed)
    print("Training bio player for env1...")
    model = train(bio_player, timeout=100)

    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score (distance): {score:.2f} +/- {std:.2f}")
