



from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def relu(x):
    return np.clip(x, a_min=0.0, a_max=None)


def identity(x):
    return x


def ensemble_selfnegate_assay_player() -> Iterable[Tuple]:
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



    t_bp = 0.003
    thr_any = 1e-5


    n_probe = 140.0



    U_SELF_HIT = 0
    U_SELF_HI = 1
    U_SELF_LO = 2
    U_NEG_HIT = 3
    U_NEG_HI = 4
    U_NEG_LO = 5


    U_SELF_DELTA = 10
    U_NEG_DELTA = 11
    U_SELF_ON = 12
    U_NEG_ON = 13

    U_E = 20
    U_E_PREV = 21
    U_DE = 22
    U_F1 = 23
    U_BP = 24
    U_ANY = 25

    U_ON_RAW = 26
    U_ON_OVER = 27
    U_ON = 28
    U_DE_D = 29

    U_SUM_SELF = 40
    U_SUM_NEG = 41
    U_COUNT_SELF = 42
    U_COUNT_NEG = 43
    U_ADD_SUM_SELF = 44
    U_ADD_SUM_NEG = 45
    U_ADD_COUNT_SELF = 46
    U_ADD_COUNT_NEG = 47

    U_EXCEED_SELF_RAW = 50
    U_EXCEED_SELF_OVER = 51
    U_EXCEED_SELF = 52
    U_PULSE_TO_NEG = 53

    U_DONE_NEG_RAW = 54
    U_DONE_NEG_OVER = 55
    U_DONE_NEG = 56

    U_DIFF = 60
    U_CHOOSE_RAW = 61
    U_CHOOSE_SCALE = 62
    U_CHOOSE_OVER = 63
    U_CHOOSE_IND = 64
    U_CHOOSE_GATE = 65
    U_PULSE_CHOOSE_SELF = 66

    Win = np.zeros((n, 2 * p + 3), dtype=np.float64)
    W = np.zeros((n, n), dtype=np.float64)
    Wout = np.zeros((1, n), dtype=np.float64)



    W[U_SELF_DELTA, U_SELF_DELTA] = 1.0
    W[U_NEG_DELTA, U_NEG_DELTA] = 1.0




    Win[U_SELF_ON, bias_idx] = 1.0
    W[U_SELF_ON, U_SELF_DELTA] = 1.0
    W[U_SELF_ON, U_NEG_DELTA] = -1.0

    Win[U_NEG_ON, bias_idx] = 0.0
    W[U_NEG_ON, U_NEG_DELTA] = 1.0
    W[U_NEG_ON, U_SELF_DELTA] = -1.0



    Win[U_E, energy_idx] = 1.0
    W[U_E_PREV, U_E] = 1.0
    Win[U_DE, energy_idx] = 1.0
    W[U_DE, U_E] = -1.0


    W[U_F1, U_DE] = 1.0
    Win[U_F1, bias_idx] = -t_bp
    W[U_DE_D, U_DE] = 1.0
    W[U_BP, U_DE] = 0.0
    W[U_BP, U_DE_D] = 1.0
    W[U_BP, U_F1] = -2.0


    W[U_ANY, U_BP] = 1.0
    Win[U_ANY, bias_idx] = -thr_any



    gain_on = 5000.0
    W[U_ON_RAW, U_ANY] = gain_on



    W[U_ON_OVER, U_ON_RAW] = -1.0
    Win[U_ON_OVER, bias_idx] = 1.0
    W[U_ON, U_ON_RAW] = 0.0
    W[U_ON, U_ON_OVER] = -1.0
    Win[U_ON, bias_idx] = 1.0



    W[U_ADD_SUM_SELF, U_BP] = 1.0
    W[U_ADD_SUM_SELF, U_SELF_ON] = 1.0
    Win[U_ADD_SUM_SELF, bias_idx] = -1.0

    W[U_ADD_SUM_NEG, U_BP] = 1.0
    W[U_ADD_SUM_NEG, U_NEG_ON] = 1.0
    Win[U_ADD_SUM_NEG, bias_idx] = -1.0



    W[U_ADD_COUNT_SELF, U_SELF_ON] = 1.0
    Win[U_ADD_COUNT_SELF, bias_idx] = 0.0

    W[U_ADD_COUNT_NEG, U_NEG_ON] = 1.0
    Win[U_ADD_COUNT_NEG, bias_idx] = 0.0

    W[U_SUM_SELF, U_SUM_SELF] = 1.0
    W[U_SUM_SELF, U_ADD_SUM_SELF] = 1.0
    W[U_SUM_NEG, U_SUM_NEG] = 1.0
    W[U_SUM_NEG, U_ADD_SUM_NEG] = 1.0

    W[U_COUNT_SELF, U_COUNT_SELF] = 1.0
    W[U_COUNT_SELF, U_ADD_COUNT_SELF] = 1.0
    W[U_COUNT_NEG, U_COUNT_NEG] = 1.0
    W[U_COUNT_NEG, U_ADD_COUNT_NEG] = 1.0






    W[U_EXCEED_SELF_RAW, U_COUNT_SELF] = 1.0
    Win[U_EXCEED_SELF_RAW, bias_idx] = -n_probe
    W[U_EXCEED_SELF_OVER, U_EXCEED_SELF_RAW] = 1.0
    Win[U_EXCEED_SELF_OVER, bias_idx] = -1.0
    W[U_EXCEED_SELF, U_EXCEED_SELF_RAW] = 1.0
    W[U_EXCEED_SELF, U_EXCEED_SELF_OVER] = -1.0
    Win[U_EXCEED_SELF, bias_idx] = 0.0


    W[U_PULSE_TO_NEG, U_EXCEED_SELF] = 1.0
    W[U_PULSE_TO_NEG, U_NEG_DELTA] = -1000.0
    W[U_NEG_DELTA, U_PULSE_TO_NEG] = 0.5



    Win[U_DONE_NEG_RAW, bias_idx] = n_probe
    W[U_DONE_NEG_RAW, U_COUNT_NEG] = -1.0
    W[U_DONE_NEG_OVER, U_DONE_NEG_RAW] = 1.0
    Win[U_DONE_NEG_OVER, bias_idx] = -1.0
    W[U_DONE_NEG, U_DONE_NEG_RAW] = 1.0
    W[U_DONE_NEG, U_DONE_NEG_OVER] = -1.0
    Win[U_DONE_NEG, bias_idx] = 0.0



    W[U_DIFF, U_SUM_SELF] = 1.0
    W[U_DIFF, U_SUM_NEG] = -1.0

    W[U_CHOOSE_RAW, U_DIFF] = 1.0

    gain_choose = 1000.0
    W[U_CHOOSE_SCALE, U_CHOOSE_RAW] = gain_choose



    W[U_CHOOSE_OVER, U_CHOOSE_SCALE] = 1.0
    Win[U_CHOOSE_OVER, bias_idx] = -1.0
    W[U_CHOOSE_IND, U_CHOOSE_SCALE] = 1.0
    W[U_CHOOSE_IND, U_CHOOSE_OVER] = -1.0
    Win[U_CHOOSE_IND, bias_idx] = 0.0


    W[U_CHOOSE_GATE, U_CHOOSE_IND] = 1.0
    W[U_CHOOSE_GATE, U_DONE_NEG] = -1.0
    Win[U_CHOOSE_GATE, bias_idx] = 0.0


    W[U_PULSE_CHOOSE_SELF, U_CHOOSE_GATE] = 1.0
    W[U_PULSE_CHOOSE_SELF, U_SELF_DELTA] = -1000.0
    W[U_SELF_DELTA, U_PULSE_CHOOSE_SELF] = 0.5



    Win[U_SELF_HIT, hit_idx] = 1.0
    W[U_SELF_HIT, U_SELF_ON] = 1.0
    Win[U_SELF_HIT, bias_idx] = -1.0

    Win[U_SELF_HI, side_right_idx] = 1.0
    Win[U_SELF_HI, bias_idx] = -(wall_target + 1.0)
    W[U_SELF_HI, U_SELF_ON] = 1.0

    Win[U_SELF_LO, side_right_idx] = -1.0
    Win[U_SELF_LO, bias_idx] = wall_target - 1.0
    W[U_SELF_LO, U_SELF_ON] = 1.0


    Win[U_NEG_HIT, hit_idx] = 1.0
    W[U_NEG_HIT, U_NEG_ON] = 1.0
    Win[U_NEG_HIT, bias_idx] = -1.0

    Win[U_NEG_HI, side_left_idx] = 1.0
    Win[U_NEG_HI, bias_idx] = -(wall_target + 1.0)
    W[U_NEG_HI, U_NEG_ON] = 1.0

    Win[U_NEG_LO, side_left_idx] = -1.0
    Win[U_NEG_LO, bias_idx] = wall_target - 1.0
    W[U_NEG_LO, U_NEG_ON] = 1.0


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

    model = train(ensemble_selfnegate_assay_player, timeout=100.0)
    np.random.seed(12345)
    score, std = evaluate(model, Bot, Environment, runs=10, debug=False)
    print(f"score={score:.6f} std={std:.6f}")
