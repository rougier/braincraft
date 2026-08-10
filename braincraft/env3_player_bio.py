# Braincraft challenge — Bio Player for Environment 3
# Copyright (C) 2026 Guanchun Li
# Released under the GNU General Public License 3

"""
Bio Player for Environment 3.

Pointwise-activation Echo State Network controller:

    X(t+1) = f(Win @ I(t) + W @ X(t))
    O(t+1) = Wout @ g(X(t+1))        (g = identity)

Every hidden activation depends only on its own preactivation; all
cross-neuron logic is carried by the connectivity matrices.

Input (131 cols): I(t) = [prox[0..63](t), colour[0..63](t),
                          hit(t), energy(t), 1].
Env3 exposes colour, but this controller does not read it — the bot
runs around the outer corridor with a reflex wall-follower and picks
up whatever source it happens to cross.

Seven hidden slots:

    0..4    reflex features (hit, proximity, safety)
    5       dtheta (clipped one-step-lagged steering command)
    6       unsigned front-block escape channel
"""

import numpy as np

if not hasattr(np, "atan2"):
    np.atan2 = np.arctan2

from bot import Bot
from environment_3 import Environment


front_gain_mag = np.radians(20.0)
step_a         = np.radians(5.0)      # actuator clip (±5°)


def _bio_indices():
    idx = {
        "hit_feat":    0,
        "prox_left":   1,
        "prox_right":  2,
        "safe_left":   3,
        "safe_right":  4,
        "dtheta":      5,
        "front_block": 6,
    }
    idx["bio_end"] = 7
    return idx


def make_activation(a, idx):
    """Per-neuron pointwise activation: clip for dtheta, relu_tanh elsewhere."""
    def f(x):
        out = np.maximum(0.0, np.tanh(x))
        out[idx["dtheta"], 0] = float(np.clip(x[idx["dtheta"], 0], -a, a))
        return out

    return f


def bio_player():
    """Build the env3 bio controller and yield a single frozen model."""

    bot = Bot()
    n = 1000
    p = bot.camera.resolution          # 64
    warmup = 0
    leak = 1.0
    g = lambda x: x

    # Env3 feeds I = [depths, colours, hit, energy, 1]. Colour and energy
    # columns are unread but the hit/bias indices sit at the full 2p+3
    # offsets.
    n_inputs = 2 * p + 3               # 131
    Win  = np.zeros((n, n_inputs))
    W    = np.zeros((n, n))
    Wout = np.zeros((1, n))

    hit_idx  = 2 * p                   # 128
    bias_idx = 2 * p + 2               # 130

    idx = _bio_indices()
    a   = step_a

    HIT_FEAT   = idx["hit_feat"]
    PROX_LEFT  = idx["prox_left"]
    PROX_RIGHT = idx["prox_right"]
    SAFE_LEFT  = idx["safe_left"]
    SAFE_RIGHT = idx["safe_right"]
    DTHETA     = idx["dtheta"]
    FB         = idx["front_block"]

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

    # Reflex features and steering readout.
    Win[HIT_FEAT,    hit_idx]        = 1.0
    Win[PROX_LEFT,   L_idx]          = 1.0
    Win[PROX_RIGHT,  R_idx]          = 1.0
    Win[SAFE_LEFT,   left_side_idx]  = -1.0
    Win[SAFE_RIGHT,  right_side_idx] = -1.0
    Win[SAFE_LEFT,   bias_idx]       = safety_target
    Win[SAFE_RIGHT,  bias_idx]       = safety_target

    Wout[0, HIT_FEAT]   = hit_turn
    Wout[0, PROX_LEFT]  = heading_gain
    Wout[0, PROX_RIGHT] = -heading_gain
    Wout[0, SAFE_LEFT]  = safety_gain_left
    Wout[0, SAFE_RIGHT] = safety_gain_right

    # Unsigned front-block: fires when the two centre proximity taps
    # exceed front_thr and drives a fixed-direction (CCW) escape turn.
    Win[FB, C1_idx]   = 1.0
    Win[FB, C2_idx]   = 1.0
    Win[FB, bias_idx] = -front_thr

    Wout[0, FB] = front_gain_mag

    # Mirror Wout into W[DTHETA, :] so dtheta(t+1) = clip(O(t), ±step_a).
    for j in range(n):
        if Wout[0, j] != 0.0:
            W[DTHETA, j] = Wout[0, j]

    f = make_activation(a, idx)
    model = Win, W, Wout, warmup, leak, f, g
    yield model


if __name__ == "__main__":
    import time
    from challenge_3 import evaluate, train

    seed = 12345
    np.random.seed(seed)
    print("Training bio player for env3...")
    model = train(bio_player, timeout=100)

    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score (distance): {score:.2f} +/- {std:.2f}")
