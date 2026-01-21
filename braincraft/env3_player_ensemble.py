# Braincraft challenge entry — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
"""
Task 3 (Valued decision) — "Ensemble" strategy

Two expert controllers (clockwise / counter-clockwise) are available.

The bot starts with the CCW expert, which reliably reaches the *left* source first.
It then uses an internal, reward-free signal derived from the bot's own energy:

  dE⁺(t) = max(0, E(t) - E(t-1))

After a short band-pass filter to reject the initial transient, the magnitude of dE⁺
distinguishes the weak source (~0.00025 net) from the strong source (~0.004 net).

If the first visited (left) source appears weak, the controller switches to the CW
expert (which reaches the right source first), otherwise it stays CCW.

This implements a task-specific "ensemble selection": increase behavioral
complexity (switch expert) only when evidence indicates the current hypothesis
("left is best") is wrong.
"""

import numpy as np

from bot import Bot
from environment_3 import Environment


def relu(x):
    return np.clip(x, a_min=0, a_max=None)


def identity(x):
    return x


def ensemble_player():
    bot = Bot()

    # Fixed parameters
    leak = 1.0
    warmup = 1
    f = relu
    g = identity

    n = 1000
    n_cam = bot.camera.resolution
    n_inp = 2 * n_cam + 3  # distances, colors, hit, energy, bias

    # Input indices (challenge_2 style)
    i_left = 0
    i_right = n_cam - 1
    i_energy = 2 * n_cam + 1
    i_bias = 2 * n_cam + 2

    # Neuron indices
    i_ccw_right = 1
    i_ccw_left = 2
    i_cw_left = 3
    i_cw_right = 4

    i_E = 100
    i_Ed = 101
    i_dE = 102

    i_f1 = 110
    i_f2 = 111
    i_bp = 112
    i_any = 113
    i_big = 114
    i_weak = 115
    i_seen_weak = 116
    i_pulse_weak = 117
    i_mode_cw = 118
    i_seen_big = 119
    i_weak_eff = 120

    W_in = np.zeros((n, n_inp))
    W = np.zeros((n, n))
    W_out = np.zeros((1, n))

    # ------------------------------------------------------------------
    # Energy-derivative detector (positive part)
    W_in[i_E, i_energy] = 1.0
    W[i_Ed, i_E] = 1.0
    W[i_dE, i_E] = 1.0
    W[i_dE, i_Ed] = -1.0

    # Triangular band-pass on dE to reject the large initial transient.
    # See env3_player_derivative_switcher.py for derivation.
    t = 0.003
    W[i_f1, i_dE] = 1.0
    W_in[i_f1, i_bias] = -t
    W[i_f2, i_dE] = 1.0
    W[i_bp, i_f1] = -2.0
    W[i_bp, i_f2] = 1.0

    # Any reward event (weak or strong)
    thr_any = 1e-5
    W[i_any, i_bp] = 1.0
    W_in[i_any, i_bias] = -thr_any

    # Strong reward event
    thr_big = 0.001
    W[i_big, i_bp] = 1.0
    W_in[i_big, i_bias] = -thr_big

    # Weak-only evidence: any - 1000*big (clips to 0 for strong)
    W[i_weak, i_any] = 1.0
    W[i_weak, i_big] = -1000.0

    # Remember if a strong reward has been observed (locks decision).
    W[i_seen_big, i_seen_big] = 1.0
    W[i_seen_big, i_big] = 1.0

    # Effective weak evidence: weak - K*seen_big (clips to 0 once strong seen)
    W[i_weak_eff, i_weak] = 1.0
    W[i_weak_eff, i_seen_big] = -1000.0

    # One-shot switching to avoid a slow interpolation between experts:
    # - i_seen_weak remembers that weak evidence was already observed.
    # - i_pulse_weak is a short pulse on the first weak observation.
    # - i_mode_cw is set to ~1 by that pulse and then held forever.
    W[i_seen_weak, i_seen_weak] = 1.0
    W[i_seen_weak, i_weak_eff] = 1.0

    W[i_pulse_weak, i_weak_eff] = 1.0
    W[i_pulse_weak, i_seen_weak] = -1000.0

    W[i_mode_cw, i_mode_cw] = 1.0
    W[i_mode_cw, i_pulse_weak] = 4167.0  # 0.00024 * 4167 ≈ 1.0

    # ------------------------------------------------------------------
    # Expert controllers (gated by mode_cw)
    #
    # CCW (reaches left source first):  +100*ReLU(sr-0.79) -50*ReLU(sl-0.88)
    # Gate: subtract mode_cw so it turns off when mode_cw ~ 1.
    W_in[i_ccw_right, i_right] = 1.0
    W_in[i_ccw_right, i_bias] = -0.79
    W[i_ccw_right, i_mode_cw] = -1.0
    W_out[0, i_ccw_right] = 100.0

    W_in[i_ccw_left, i_left] = 1.0
    W_in[i_ccw_left, i_bias] = -0.88
    W[i_ccw_left, i_mode_cw] = -1.0
    W_out[0, i_ccw_left] = -50.0

    # Switch expert (escape to the other side):
    #   -100*ReLU(sl-0.88) +50*ReLU(sr-0.79)
    #
    # This is intentionally *not* the naive mirrored CW controller: it keeps
    # the same wall-avoidance thresholds as the CCW expert, but flips the
    # steering signs. Empirically this avoids the wall-hit burst that happens
    # when switching direction while still inside the left corridor.
    #
    # Gate: add (mode_cw - 1) so it is off when mode_cw=0 and on when ~1.
    W_in[i_cw_left, i_left] = 1.0
    W_in[i_cw_left, i_bias] = -(0.88 + 1.0)
    W[i_cw_left, i_mode_cw] = 1.0
    W_out[0, i_cw_left] = -100.0

    W_in[i_cw_right, i_right] = 1.0
    W_in[i_cw_right, i_bias] = -(0.79 + 1.0)
    W[i_cw_right, i_mode_cw] = 1.0
    W_out[0, i_cw_right] = 50.0

    model = (W_in, W, W_out, warmup, leak, f, g)
    yield model
    return model


if __name__ == "__main__":
    import time
    from challenge_2 import train, evaluate

    seed = 12345

    np.random.seed(seed)
    print("Starting training for 100 seconds (user time)")
    model = train(ensemble_player, timeout=100)

    np.random.seed(seed)
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, runs=10, debug=False)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")
