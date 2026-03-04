

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def relu(x):
    return np.clip(x, a_min=0.0, a_max=None)


def identity(x):
    return x


def ensemble_counterfactual_twin_delib_v2_player() -> Iterable[Tuple]:
    n = 1000
    p = 64
    warmup = 1
    leak = 1.0

    hit_idx = 2 * p + 0
    energy_idx = 2 * p + 1
    bias_idx = 2 * p + 2

    side_right_idx = int(np.clip(52, 0, p - 1))
    side_left_idx = (p - 1) - side_right_idx

    hit_turn = 5.0
    wall_gain = 0.40
    wall_target = 0.65


    n_probe_self = 650.0
    n_probe_neg = 650.0


    t_bp = 0.003
    g_refill = 650.0
    g_drop = 220.0
    g_hit = 0.8
    g_side = 0.8
    thr_u = 0.02


    alpha_q = 0.18
    beta_sigma = 0.12
    k_uncertainty = 0.55
    theta_commit = 0.02
    choose_gain = 180.0

    gate_switch = 1.2
    gate_choose = 1.2



    U_SELF_HIT = 0
    U_SELF_HI = 1
    U_SELF_LO = 2
    U_NEG_HIT = 3
    U_NEG_HI = 4
    U_NEG_LO = 5

    U_E = 20
    U_E_PREV = 21
    U_RISE = 22
    U_DROP = 23
    U_F1 = 24
    U_BP = 25

    U_MODE_NEG = 30
    U_MODE_NEG_OVER = 31
    U_SELF_ON = 32
    U_NEG_ON = 33

    U_COUNT_SELF = 40
    U_COUNT_NEG = 41
    U_SELF_DONE_RAW = 42
    U_SELF_DONE_OVER = 43
    U_SELF_DONE = 44
    U_SELF_DONE_SEEN = 45
    U_SELF_DONE_PULSE = 46
    U_NEG_DONE_RAW = 47
    U_NEG_DONE_OVER = 48
    U_NEG_DONE = 49
    U_NEG_DONE_SEEN = 50
    U_NEG_DONE_PULSE = 51

    U_U_SELF_RAW = 60
    U_U_NEG_RAW = 61
    U_U_SELF_ACT = 62
    U_U_NEG_ACT = 63

    U_Q_SELF = 70
    U_Q_NEG = 71
    U_ERR_SELF_POS = 72
    U_ERR_SELF_NEG = 73
    U_ERR_NEG_POS = 74
    U_ERR_NEG_NEG = 75
    U_SIGMA_SELF = 76
    U_SIGMA_NEG = 77

    U_A_RAW = 80
    U_A_POS = 81
    U_A_SCALE = 82
    U_A_OVER = 83
    U_A_IND = 84
    U_CHOOSE_SELF_PULSE = 85

    Win = np.zeros((n, 2 * p + 3), dtype=np.float64)
    W = np.zeros((n, n), dtype=np.float64)
    Wout = np.zeros((1, n), dtype=np.float64)



    Win[U_E, energy_idx] = 1.0
    W[U_E_PREV, U_E] = 1.0

    Win[U_RISE, energy_idx] = 1.0
    W[U_RISE, U_E] = -1.0

    W[U_DROP, U_E] = 1.0
    Win[U_DROP, energy_idx] = -1.0

    W[U_F1, U_RISE] = 1.0
    Win[U_F1, bias_idx] = -t_bp
    W[U_BP, U_RISE] = 1.0
    W[U_BP, U_F1] = -2.0



    W[U_MODE_NEG, U_MODE_NEG] = 1.0
    W[U_MODE_NEG, U_SELF_DONE_PULSE] = gate_switch
    W[U_MODE_NEG, U_CHOOSE_SELF_PULSE] = -gate_choose
    W[U_MODE_NEG, U_MODE_NEG_OVER] = -1.0

    W[U_MODE_NEG_OVER, U_MODE_NEG] = 1.0
    Win[U_MODE_NEG_OVER, bias_idx] = -1.0

    Win[U_SELF_ON, bias_idx] = 1.0
    W[U_SELF_ON, U_MODE_NEG] = -1.0
    W[U_NEG_ON, U_MODE_NEG] = 1.0

    W[U_COUNT_SELF, U_COUNT_SELF] = 1.0
    W[U_COUNT_SELF, U_SELF_ON] = 1.0
    W[U_COUNT_NEG, U_COUNT_NEG] = 1.0
    W[U_COUNT_NEG, U_NEG_ON] = 1.0

    W[U_SELF_DONE_RAW, U_COUNT_SELF] = 1.0
    Win[U_SELF_DONE_RAW, bias_idx] = -n_probe_self
    W[U_SELF_DONE_OVER, U_SELF_DONE_RAW] = 1.0
    Win[U_SELF_DONE_OVER, bias_idx] = -1.0
    W[U_SELF_DONE, U_SELF_DONE_RAW] = 1.0
    W[U_SELF_DONE, U_SELF_DONE_OVER] = -1.0
    W[U_SELF_DONE_SEEN, U_SELF_DONE_SEEN] = 1.0
    W[U_SELF_DONE_SEEN, U_SELF_DONE] = 1.0
    W[U_SELF_DONE_PULSE, U_SELF_DONE] = 1.0
    W[U_SELF_DONE_PULSE, U_SELF_DONE_SEEN] = -1000.0

    W[U_NEG_DONE_RAW, U_COUNT_NEG] = 1.0
    Win[U_NEG_DONE_RAW, bias_idx] = -n_probe_neg
    W[U_NEG_DONE_OVER, U_NEG_DONE_RAW] = 1.0
    Win[U_NEG_DONE_OVER, bias_idx] = -1.0
    W[U_NEG_DONE, U_NEG_DONE_RAW] = 1.0
    W[U_NEG_DONE, U_NEG_DONE_OVER] = -1.0
    W[U_NEG_DONE_SEEN, U_NEG_DONE_SEEN] = 1.0
    W[U_NEG_DONE_SEEN, U_NEG_DONE] = 1.0
    W[U_NEG_DONE_PULSE, U_NEG_DONE] = 1.0
    W[U_NEG_DONE_PULSE, U_NEG_DONE_SEEN] = -1000.0




    Win[U_U_SELF_RAW, side_right_idx] = g_side
    W[U_U_SELF_RAW, U_BP] = g_refill
    W[U_U_SELF_RAW, U_DROP] = -g_drop
    Win[U_U_SELF_RAW, hit_idx] = -g_hit
    Win[U_U_SELF_RAW, bias_idx] = -thr_u

    Win[U_U_NEG_RAW, side_left_idx] = g_side
    W[U_U_NEG_RAW, U_BP] = g_refill
    W[U_U_NEG_RAW, U_DROP] = -g_drop
    Win[U_U_NEG_RAW, hit_idx] = -g_hit
    Win[U_U_NEG_RAW, bias_idx] = -thr_u


    W[U_U_SELF_ACT, U_U_SELF_RAW] = 1.0
    W[U_U_SELF_ACT, U_SELF_ON] = 1.0
    Win[U_U_SELF_ACT, bias_idx] = -1.0

    W[U_U_NEG_ACT, U_U_NEG_RAW] = 1.0
    W[U_U_NEG_ACT, U_NEG_ON] = 1.0
    Win[U_U_NEG_ACT, bias_idx] = -1.0


    W[U_Q_SELF, U_Q_SELF] = 1.0 - alpha_q
    W[U_Q_SELF, U_U_SELF_ACT] = alpha_q
    W[U_Q_NEG, U_Q_NEG] = 1.0 - alpha_q
    W[U_Q_NEG, U_U_NEG_ACT] = alpha_q


    W[U_ERR_SELF_POS, U_U_SELF_ACT] = 1.0
    W[U_ERR_SELF_POS, U_Q_SELF] = -1.0
    W[U_ERR_SELF_NEG, U_Q_SELF] = 1.0
    W[U_ERR_SELF_NEG, U_U_SELF_ACT] = -1.0

    W[U_ERR_NEG_POS, U_U_NEG_ACT] = 1.0
    W[U_ERR_NEG_POS, U_Q_NEG] = -1.0
    W[U_ERR_NEG_NEG, U_Q_NEG] = 1.0
    W[U_ERR_NEG_NEG, U_U_NEG_ACT] = -1.0


    W[U_SIGMA_SELF, U_SIGMA_SELF] = 1.0 - beta_sigma
    W[U_SIGMA_SELF, U_ERR_SELF_POS] = beta_sigma
    W[U_SIGMA_SELF, U_ERR_SELF_NEG] = beta_sigma

    W[U_SIGMA_NEG, U_SIGMA_NEG] = 1.0 - beta_sigma
    W[U_SIGMA_NEG, U_ERR_NEG_POS] = beta_sigma
    W[U_SIGMA_NEG, U_ERR_NEG_NEG] = beta_sigma


    W[U_A_RAW, U_Q_SELF] = 1.0
    W[U_A_RAW, U_Q_NEG] = -1.0
    W[U_A_RAW, U_SIGMA_SELF] = -k_uncertainty
    W[U_A_RAW, U_SIGMA_NEG] = -k_uncertainty
    Win[U_A_RAW, bias_idx] = -theta_commit

    W[U_A_POS, U_A_RAW] = 1.0
    W[U_A_SCALE, U_A_POS] = choose_gain
    W[U_A_OVER, U_A_SCALE] = 1.0
    Win[U_A_OVER, bias_idx] = -1.0
    W[U_A_IND, U_A_SCALE] = 1.0
    W[U_A_IND, U_A_OVER] = -1.0


    W[U_CHOOSE_SELF_PULSE, U_A_IND] = 1.0
    W[U_CHOOSE_SELF_PULSE, U_NEG_DONE_PULSE] = 1.0
    Win[U_CHOOSE_SELF_PULSE, bias_idx] = -1.0



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

    f = relu
    g = identity
    yield Win, W, Wout, warmup, leak, f, g


if __name__ == "__main__":
    from pathlib import Path
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "braincraft" / "braincraft"))

    from challenge_3 import train, evaluate
    from bot import Bot
    from environment_3 import Environment

    np.random.seed(12345)
    model = train(ensemble_counterfactual_twin_delib_v2_player, timeout=100.0)
    score, std = evaluate(model, Bot, Environment, runs=10, seed=12345, debug=False)
    print(f"score={score:.6f} std={std:.6f}")
