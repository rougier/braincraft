# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
"""
High-scoring Task 3 player.

This is a minimal, hand-crafted controller that reliably reaches and cycles
over energy sources using wall-following plus a collision reflex.
"""

import time
import numpy as np

from bot import Bot
from environment_3 import Environment


def identity(x):
    return x


def wallfollow_player():
    """Build and return a strong Task 3 model (single yield)."""

    bot = Bot()

    # Fixed parameters
    n = 1000
    p = bot.camera.resolution
    warmup = 0
    leak = 1.0
    f = identity
    g = identity

    # Task 3 uses depth + color sensors (2*p) plus (hit, energy, bias).
    Win = np.zeros((n, 2 * p + 3))
    W = np.zeros((n, n))
    Wout = np.zeros((1, n))

    hit_idx = 2 * p
    bias_idx = 2 * p + 2

    # Use a right-side depth ray (not the extreme ray) for stable wall-following.
    side_idx = 52

    # Internal state: X0=hit, X1=bias, X2=side proximity (1 - depth[side_idx]).
    Win[0, hit_idx] = 1.0
    Win[1, bias_idx] = 1.0
    Win[2, side_idx] = 1.0

    hit_turn = 5.0
    wall_gain = 0.40
    wall_target = 0.65

    # O = hit_turn*hit + wall_gain*(side - wall_target)
    Wout[0, 0] = hit_turn
    Wout[0, 1] = -wall_gain * wall_target
    Wout[0, 2] = wall_gain

    model = Win, W, Wout, warmup, leak, f, g
    yield model


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from challenge_2 import train, evaluate

    seed = 12345

    np.random.seed(seed)
    print("Starting training for 100 seconds (user time)")
    model = train(wallfollow_player, timeout=100)

    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")
