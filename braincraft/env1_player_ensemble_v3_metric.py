# Braincraft challenge entry — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def relu(x):
    return np.clip(x, a_min=0.0, a_max=None)


def identity(x):
    return x


def ensemble_v3_metric_env1_player() -> Iterable[Tuple]:
    n = 1000
    p = 64
    warmup = 1
    leak = 1.0

    n_inp = p + 3
    hit_idx = p + 0
    energy_idx = p + 1
    bias_idx = p + 2

    side_right_idx = int(np.clip(52, 0, p - 1))
    side_left_idx = (p - 1) - side_right_idx

    wall_target = 0.65
    wall_gain = 0.40
    hit_turn = 5.0

    alpha_sigma = 0.12
    eta = 0.06
    r_star = 0.30
    eps0 = 0.55
    eps_decay = 0.04
    hit_gain = 3.0
    mode_gain = 0.30
    mode_theta = 0.03

    Win = np.zeros((n, n_inp), dtype=np.float64)
    W = np.zeros((n, n), dtype=np.float64)
    Wout = np.zeros((1, n), dtype=np.float64)

    U_SELF_HIT = 0
    U_SELF_HI = 1
    U_SELF_LO = 2
    U_NEG_HIT = 3
    U_NEG_HI = 4
    U_NEG_LO = 5

    U_ERR_R_HI = 30
    U_ERR_R_LO = 31
    U_ERR_R = 32
    U_ERR_L_HI = 33
    U_ERR_L_LO = 34
    U_ERR_L = 35

    U_E = 20
    U_E_PREV = 21
    U_RISE = 22
    U_DROP = 23
    U_F1 = 24
    U_BP = 25

    U_D_SELF = 40
    U_D_NEG = 41
    U_SIG_SELF = 42
    U_SIG_NEG = 43

    U_DIST_SELF_POS = 44
    U_DIST_SELF_NEG = 45
    U_DIST_SELF = 46
    U_DIST_NEG_POS = 47
    U_DIST_NEG_NEG = 48
    U_DIST_NEG = 49

    U_EPS = 50
    U_H_SELF_RAW = 51
    U_H_SELF_SCALE = 52
    U_H_SELF_OVER = 53
    U_H_SELF = 54
    U_H_NEG_RAW = 55
    U_H_NEG_SCALE = 56
    U_H_NEG_OVER = 57
    U_H_NEG = 58
    U_R = 60

    U_SCORE_SELF = 70
    U_SCORE_NEG = 71
    U_DIFF_POS = 72
    U_DIFF_NEG = 73
    U_MODE_NEG = 74
    U_MODE_NEG_OVER = 75
    U_SELF_ON = 76
    U_NEG_ON = 77

    Win[U_E, energy_idx] = 1.0
    W[U_E_PREV, U_E] = 1.0

    Win[U_RISE, energy_idx] = 1.0
    W[U_RISE, U_E] = -1.0

    W[U_DROP, U_E] = 1.0
    Win[U_DROP, energy_idx] = -1.0

    W[U_F1, U_RISE] = 1.0
    Win[U_F1, bias_idx] = -0.003
    W[U_BP, U_RISE] = 1.0
    W[U_BP, U_F1] = -2.0

    Win[U_ERR_R_HI, side_right_idx] = 1.0
    Win[U_ERR_R_HI, bias_idx] = -wall_target
    Win[U_ERR_R_LO, side_right_idx] = -1.0
    Win[U_ERR_R_LO, bias_idx] = wall_target
    W[U_ERR_R, U_ERR_R_HI] = 1.0
    W[U_ERR_R, U_ERR_R_LO] = 1.0

    Win[U_ERR_L_HI, side_left_idx] = 1.0
    Win[U_ERR_L_HI, bias_idx] = -wall_target
    Win[U_ERR_L_LO, side_left_idx] = -1.0
    Win[U_ERR_L_LO, bias_idx] = wall_target
    W[U_ERR_L, U_ERR_L_HI] = 1.0
    W[U_ERR_L, U_ERR_L_LO] = 1.0

    W[U_D_SELF, U_ERR_R] = 1.40
    W[U_D_SELF, U_DROP] = 120.0
    W[U_D_SELF, U_BP] = -180.0
    Win[U_D_SELF, hit_idx] = 0.90
    Win[U_D_SELF, bias_idx] = -0.02

    W[U_D_NEG, U_ERR_L] = 1.40
    W[U_D_NEG, U_DROP] = 120.0
    W[U_D_NEG, U_BP] = -180.0
    Win[U_D_NEG, hit_idx] = 0.90
    Win[U_D_NEG, bias_idx] = -0.02

    W[U_SIG_SELF, U_SIG_SELF] = 1.0 - alpha_sigma
    W[U_SIG_SELF, U_D_SELF] = alpha_sigma
    W[U_SIG_NEG, U_SIG_NEG] = 1.0 - alpha_sigma
    W[U_SIG_NEG, U_D_NEG] = alpha_sigma

    W[U_DIST_SELF_POS, U_D_SELF] = 1.0
    W[U_DIST_SELF_POS, U_SIG_SELF] = -1.0
    W[U_DIST_SELF_NEG, U_SIG_SELF] = 1.0
    W[U_DIST_SELF_NEG, U_D_SELF] = -1.0
    W[U_DIST_SELF, U_DIST_SELF_POS] = 1.0
    W[U_DIST_SELF, U_DIST_SELF_NEG] = 1.0

    W[U_DIST_NEG_POS, U_D_NEG] = 1.0
    W[U_DIST_NEG_POS, U_SIG_NEG] = -1.0
    W[U_DIST_NEG_NEG, U_SIG_NEG] = 1.0
    W[U_DIST_NEG_NEG, U_D_NEG] = -1.0
    W[U_DIST_NEG, U_DIST_NEG_POS] = 1.0
    W[U_DIST_NEG, U_DIST_NEG_NEG] = 1.0

    W[U_EPS, U_EPS] = 1.0 - eps_decay
    W[U_EPS, U_R] = -eta
    Win[U_EPS, bias_idx] = eps_decay * eps0 + eta * r_star

    W[U_H_SELF_RAW, U_EPS] = 1.0
    W[U_H_SELF_RAW, U_DIST_SELF] = -1.0
    W[U_H_SELF_SCALE, U_H_SELF_RAW] = hit_gain
    W[U_H_SELF_OVER, U_H_SELF_SCALE] = 1.0
    Win[U_H_SELF_OVER, bias_idx] = -1.0
    W[U_H_SELF, U_H_SELF_SCALE] = 1.0
    W[U_H_SELF, U_H_SELF_OVER] = -1.0

    W[U_H_NEG_RAW, U_EPS] = 1.0
    W[U_H_NEG_RAW, U_DIST_NEG] = -1.0
    W[U_H_NEG_SCALE, U_H_NEG_RAW] = hit_gain
    W[U_H_NEG_OVER, U_H_NEG_SCALE] = 1.0
    Win[U_H_NEG_OVER, bias_idx] = -1.0
    W[U_H_NEG, U_H_NEG_SCALE] = 1.0
    W[U_H_NEG, U_H_NEG_OVER] = -1.0

    W[U_R, U_R] = 0.90
    W[U_R, U_H_SELF] = 0.05
    W[U_R, U_H_NEG] = 0.05

    W[U_SCORE_SELF, U_H_SELF] = 0.90
    W[U_SCORE_SELF, U_BP] = 0.20
    W[U_SCORE_SELF, U_DROP] = -0.15

    W[U_SCORE_NEG, U_H_NEG] = 0.90
    W[U_SCORE_NEG, U_BP] = 0.20
    W[U_SCORE_NEG, U_DROP] = -0.15

    W[U_DIFF_POS, U_SCORE_NEG] = 1.0
    W[U_DIFF_POS, U_SCORE_SELF] = -1.0
    Win[U_DIFF_POS, bias_idx] = -mode_theta

    W[U_DIFF_NEG, U_SCORE_SELF] = 1.0
    W[U_DIFF_NEG, U_SCORE_NEG] = -1.0
    Win[U_DIFF_NEG, bias_idx] = -mode_theta

    W[U_MODE_NEG, U_MODE_NEG] = 1.0
    W[U_MODE_NEG, U_DIFF_POS] = mode_gain
    W[U_MODE_NEG, U_DIFF_NEG] = -mode_gain
    W[U_MODE_NEG, U_MODE_NEG_OVER] = -1.0
    W[U_MODE_NEG_OVER, U_MODE_NEG] = 1.0
    Win[U_MODE_NEG_OVER, bias_idx] = -1.0

    Win[U_SELF_ON, bias_idx] = 1.0
    W[U_SELF_ON, U_MODE_NEG] = -1.0
    W[U_NEG_ON, U_MODE_NEG] = 1.0

    Win[U_SELF_HIT, hit_idx] = 1.0
    W[U_SELF_HIT, U_SELF_ON] = 1.0
    Win[U_SELF_HIT, bias_idx] = -1.0

    Win[U_SELF_HI, side_right_idx] = 1.0
    W[U_SELF_HI, U_SELF_ON] = 1.0
    Win[U_SELF_HI, bias_idx] = -(wall_target + 1.0)

    Win[U_SELF_LO, side_right_idx] = -1.0
    W[U_SELF_LO, U_SELF_ON] = 1.0
    Win[U_SELF_LO, bias_idx] = wall_target - 1.0

    Win[U_NEG_HIT, hit_idx] = 1.0
    W[U_NEG_HIT, U_NEG_ON] = 1.0
    Win[U_NEG_HIT, bias_idx] = -1.0

    Win[U_NEG_HI, side_left_idx] = 1.0
    W[U_NEG_HI, U_NEG_ON] = 1.0
    Win[U_NEG_HI, bias_idx] = -(wall_target + 1.0)

    Win[U_NEG_LO, side_left_idx] = -1.0
    W[U_NEG_LO, U_NEG_ON] = 1.0
    Win[U_NEG_LO, bias_idx] = wall_target - 1.0

    Wout[0, U_SELF_HIT] = +hit_turn
    Wout[0, U_SELF_HI] = +wall_gain
    Wout[0, U_SELF_LO] = -wall_gain
    Wout[0, U_NEG_HIT] = -hit_turn
    Wout[0, U_NEG_HI] = -wall_gain
    Wout[0, U_NEG_LO] = +wall_gain

    # Tuned stabilization: keep the strong self branch always active.
    W[U_MODE_NEG, :] = 0.0
    Win[U_MODE_NEG, :] = 0.0
    W[U_NEG_ON, :] = 0.0
    Win[U_NEG_ON, :] = 0.0
    Win[U_SELF_ON, bias_idx] = 1.0
    Wout[0, U_NEG_HIT] = 0.0
    Wout[0, U_NEG_HI] = 0.0
    Wout[0, U_NEG_LO] = 0.0

    f = relu
    g = identity
    yield Win, W, Wout, warmup, leak, f, g


if __name__ == "__main__":
    from pathlib import Path
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "braincraft" / "braincraft"))

    from challenge_1 import train, evaluate
    from bot import Bot
    from environment_1 import Environment

    np.random.seed(12345)
    model = train(ensemble_v3_metric_env1_player, timeout=100.0)
    score, std = evaluate(model, Bot, Environment, runs=10, seed=12345, debug=False)
    print(f"score={score:.6f} std={std:.6f}")
