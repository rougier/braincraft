



from __future__ import annotations

import time
from typing import Iterable, Tuple

import numpy as np

from bot import Bot
from environment_3 import Environment


def relu(x):
    return np.clip(x, a_min=0.0, a_max=None)


def identity(x):
    return x


def build_ensemble_selfnegate_model(
    *,
    wall_target: float = 0.65,
    wall_gain: float = 0.40,
    hit_turn: float = 5.0,
    t_bp: float = 0.003,
    thr_any: float = 1e-5,
    thr_big: float = 1e-3,
    gate_gain: float = 4167.0,
    side_right_idx: int = 52,
) -> Tuple:


    n = 1000
    p = 64
    warmup = 1
    leak = 1.0







    hit_idx = 2 * p + 0
    energy_idx = 2 * p + 1
    bias_idx = 2 * p + 2

    side_right_idx = int(np.clip(int(side_right_idx), 0, p - 1))
    side_left_idx = (p - 1) - side_right_idx


    hit_turn = float(hit_turn)
    wall_gain = float(wall_gain)
    wall_target = float(wall_target)









    t_bp = float(t_bp)
    thr_any = float(thr_any)
    thr_big = float(thr_big)
    gate_gain = float(gate_gain)


    U_SELF_HIT = 0
    U_SELF_HI = 1
    U_SELF_LO = 2
    U_NEG_HIT = 3
    U_NEG_HI = 4
    U_NEG_LO = 5

    U_E = 20
    U_E_PREV = 21
    U_DE = 22
    U_F1 = 23
    U_BP = 24
    U_ANY = 25
    U_BIG = 26
    U_SEEN_BIG = 27
    U_WEAK = 28
    U_WEAK_EFF = 29
    U_SEEN_WEAK = 30
    U_PULSE_WEAK = 31
    U_MODE = 32

    Win = np.zeros((n, 2 * p + 3), dtype=np.float64)
    W = np.zeros((n, n), dtype=np.float64)
    Wout = np.zeros((1, n), dtype=np.float64)





    Win[U_E, energy_idx] = 1.0
    W[U_E_PREV, U_E] = 1.0


    Win[U_DE, energy_idx] = 1.0
    W[U_DE, U_E] = -1.0

    W[U_F1, U_DE] = 1.0
    Win[U_F1, bias_idx] = -t_bp

    W[U_BP, U_DE] = 1.0
    W[U_BP, U_F1] = -2.0


    W[U_ANY, U_BP] = 1.0
    Win[U_ANY, bias_idx] = -thr_any

    W[U_BIG, U_BP] = 1.0
    Win[U_BIG, bias_idx] = -thr_big


    W[U_SEEN_BIG, U_SEEN_BIG] = 1.0
    W[U_SEEN_BIG, U_BIG] = 1.0


    W[U_WEAK, U_ANY] = 1.0
    W[U_WEAK, U_BIG] = -1000.0

    W[U_WEAK_EFF, U_WEAK] = 1.0
    W[U_WEAK_EFF, U_SEEN_BIG] = -1000.0


    W[U_SEEN_WEAK, U_SEEN_WEAK] = 1.0
    W[U_SEEN_WEAK, U_WEAK_EFF] = 1.0

    W[U_PULSE_WEAK, U_WEAK_EFF] = 1.0
    W[U_PULSE_WEAK, U_SEEN_WEAK] = -1000.0


    W[U_MODE, U_MODE] = 1.0
    W[U_MODE, U_PULSE_WEAK] = gate_gain






    Win[U_SELF_HIT, hit_idx] = 1.0
    W[U_SELF_HIT, U_MODE] = -1.0

    Win[U_SELF_HI, side_right_idx] = 1.0
    Win[U_SELF_HI, bias_idx] = -wall_target
    W[U_SELF_HI, U_MODE] = -1.0

    Win[U_SELF_LO, side_right_idx] = -1.0
    Win[U_SELF_LO, bias_idx] = wall_target
    W[U_SELF_LO, U_MODE] = -1.0




    Win[U_NEG_HIT, hit_idx] = 1.0
    Win[U_NEG_HIT, bias_idx] = -1.0
    W[U_NEG_HIT, U_MODE] = 1.0

    Win[U_NEG_HI, side_left_idx] = 1.0
    Win[U_NEG_HI, bias_idx] = -(wall_target + 1.0)
    W[U_NEG_HI, U_MODE] = 1.0

    Win[U_NEG_LO, side_left_idx] = -1.0
    Win[U_NEG_LO, bias_idx] = wall_target - 1.0
    W[U_NEG_LO, U_MODE] = 1.0


    Wout[0, U_SELF_HIT] = +hit_turn
    Wout[0, U_SELF_HI] = +wall_gain
    Wout[0, U_SELF_LO] = -wall_gain

    Wout[0, U_NEG_HIT] = -hit_turn
    Wout[0, U_NEG_HI] = -wall_gain
    Wout[0, U_NEG_LO] = +wall_gain

    f = relu
    g = identity

    return Win, W, Wout, warmup, leak, f, g


def ensemble_selfnegate_player() -> Iterable[Tuple]:
    yield build_ensemble_selfnegate_model()



if __name__ == "__main__":
    from challenge_3 import train, evaluate

    seed = 12345
    np.random.seed(seed)

    print("Starting training for 100 seconds (user time)")
    model = train(ensemble_selfnegate_player, timeout=100)

    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")
