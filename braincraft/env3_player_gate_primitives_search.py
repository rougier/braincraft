# Braincraft challenge entry — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
"""
Task 3 (Valued decision) — Gate + Primitive Search

Novel path: keep the energy-derivative gate, but optimize the *steering primitives*
of each expert (CCW / CW) using a fast CEM search during the 100s training window.

FROZEN SUBMISSION SNAPSHOT (best so far in this repo): 15.13 ± 0.11 @ seed=12345.

Training uses an internal simulator that raycasts only a small subset of camera rays
to evaluate candidate primitive parameters quickly. The final yielded model is a
sparse ESN that implements the best parameters found.
"""

import time
import numpy as np

from bot import Bot
from environment_3 import Environment
from challenge_2 import evaluate


def relu(x):
    return np.clip(x, a_min=0.0, a_max=None)


def identity(x):
    return x


RAY_IDX = np.array([0, 16, 32, 48, 63], dtype=int)


def _clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


def build_model(params):
    """
    params = [thr[0..k-1], w_ccw[0..k-1], w_cw[0..k-1]]
    where k=len(RAY_IDX), thr in [0.2, 0.99], weights in [-200, 200].
    """
    k = int(RAY_IDX.size)
    thr = np.array(params[:k], dtype=float)
    w_ccw = np.array(params[k:2 * k], dtype=float)
    w_cw = np.array(params[2 * k:3 * k], dtype=float)

    thr = np.clip(thr, 0.2, 0.99)
    w_ccw = np.clip(w_ccw, -200.0, 200.0)
    w_cw = np.clip(w_cw, -200.0, 200.0)

    bot = Bot()
    n = 1000
    n_cam = bot.camera.resolution
    n_inp = 2 * n_cam + 3  # distances, colors, hit, energy, bias

    i_energy = 2 * n_cam + 1
    i_bias = 2 * n_cam + 2

    # Fixed gate hyperparams (kept constant; we search primitives first)
    t_bp = 0.003
    thr_any = 1e-5
    thr_big = 1e-3
    gate_gain = 4167.0
    thr_time = 0.5
    clock_step = 0.001

    # Indices
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
    i_clock = 121
    i_probe = 122
    i_force = 123

    # Primitive units
    i_ccw0 = 1
    i_cw0 = i_ccw0 + k

    W_in = np.zeros((n, n_inp))
    W = np.zeros((n, n))
    W_out = np.zeros((1, n))

    # ------------------------------------------------------------
    # Energy derivative and band-pass
    W_in[i_E, i_energy] = 1.0
    W[i_Ed, i_E] = 1.0
    W[i_dE, i_E] = 1.0
    W[i_dE, i_Ed] = -1.0

    W[i_f1, i_dE] = 1.0
    W_in[i_f1, i_bias] = -t_bp
    W[i_f2, i_dE] = 1.0
    W[i_bp, i_f1] = -2.0
    W[i_bp, i_f2] = 1.0

    # Any / big / weak evidence
    W[i_any, i_bp] = 1.0
    W_in[i_any, i_bias] = -thr_any

    W[i_big, i_bp] = 1.0
    W_in[i_big, i_bias] = -thr_big

    W[i_weak, i_any] = 1.0
    W[i_weak, i_big] = -1000.0

    W[i_seen_big, i_seen_big] = 1.0
    W[i_seen_big, i_big] = 1.0

    W[i_weak_eff, i_weak] = 1.0
    W[i_weak_eff, i_seen_big] = -1000.0

    W[i_seen_weak, i_seen_weak] = 1.0
    W[i_seen_weak, i_weak_eff] = 1.0

    W[i_pulse_weak, i_weak_eff] = 1.0
    W[i_pulse_weak, i_seen_weak] = -1000.0

    # Timed probe -> force switch if no decision
    W[i_clock, i_clock] = 1.0
    W_in[i_clock, i_bias] = clock_step

    W[i_probe, i_clock] = 1.0
    W_in[i_probe, i_bias] = -thr_time

    W[i_force, i_probe] = 1.0
    W[i_force, i_seen_big] = -1000.0
    W[i_force, i_seen_weak] = -1000.0

    # Mode latch
    W[i_mode_cw, i_mode_cw] = 1.0
    W[i_mode_cw, i_pulse_weak] = gate_gain
    W[i_mode_cw, i_force] = gate_gain

    # ------------------------------------------------------------
    # Primitives (shared thresholds, separate weights)
    for j, idx in enumerate(RAY_IDX):
        u_ccw = i_ccw0 + j
        u_cw = i_cw0 + j

        # CCW on when mode_cw ~ 0, off when ~1
        W_in[u_ccw, int(idx)] = 1.0
        W_in[u_ccw, i_bias] = -float(thr[j])
        W[u_ccw, i_mode_cw] = -1.0
        W_out[0, u_ccw] = float(w_ccw[j])

        # CW on when mode_cw ~ 1, off when ~0
        W_in[u_cw, int(idx)] = 1.0
        W_in[u_cw, i_bias] = -(float(thr[j]) + 1.0)
        W[u_cw, i_mode_cw] = 1.0
        W_out[0, u_cw] = float(w_cw[j])

    warmup = 1
    leak = 1.0
    f = relu
    g = identity
    return (W_in, W, W_out, warmup, leak, f, g)


def rollout_distance(params, seed, max_steps=4000):
    """Fast rollout using only the rays in RAY_IDX (no full camera update)."""
    np.random.seed(seed)
    bot = Bot()
    env = Environment()
    cam = bot.camera

    k = int(RAY_IDX.size)
    thr = np.clip(np.array(params[:k], dtype=float), 0.2, 0.99)
    w_ccw = np.clip(np.array(params[k:2 * k], dtype=float), -200.0, 200.0)
    w_cw = np.clip(np.array(params[2 * k:3 * k], dtype=float), -200.0, 200.0)

    # Gate constants (must match build_model)
    t_bp = 0.003
    thr_any = 1e-5
    thr_big = 1e-3
    gate_gain = 4167.0
    thr_time = 0.5
    clock_step = 0.001

    # Precompute ray offsets (match Camera.update)
    D = 0.25
    Wp = 2 * D * np.tan(np.radians(cam.fov) / 2)
    X = Wp / 2 * np.linspace(+1, -1, cam.resolution, endpoint=True)
    offsets = np.arctan2(X[RAY_IDX], D).astype(float)

    def proximity(origin, direction):
        end, cell, face, steps = cam.raycast(origin, direction, env.world)
        if end is None:
            return 0.0
        d = float(np.linalg.norm(np.asarray(origin, dtype=float) - end))
        return 1.0 - d

    prox = np.array(
        [proximity(bot.position, bot.direction + off) for off in offsets],
        dtype=float,
    )

    # Gate state (matches the sparse ESN timing: 1-step delays everywhere)
    x_E = 0.0
    x_Ed = 0.0
    seen_big = 0.0
    seen_weak = 0.0
    mode_cw = 0.0
    clock = 0.0

    distance = 0.0
    warmup = 1
    rad5 = float(np.radians(5))

    for t in range(max_steps):
        if bot.energy <= 0:
            break

        energy = float(bot.energy)

        # Expert action (gated primitives)
        m = mode_cw
        ccw_feat = relu(prox - thr - m)
        cw_feat = relu(prox + m - (thr + 1.0))
        O = float(np.dot(w_ccw, ccw_feat) + np.dot(w_cw, cw_feat))

        # Gate dynamics (all computed from previous state + current input)
        dE = relu(x_E - x_Ed)
        f1 = relu(dE - t_bp)
        bp = relu(-2.0 * f1 + dE)
        any_r = relu(bp - thr_any)
        big_r = relu(bp - thr_big)
        weak = relu(any_r - 1000.0 * big_r)

        seen_big_new = relu(seen_big + big_r)
        weak_eff = relu(weak - 1000.0 * seen_big)
        seen_weak_new = relu(seen_weak + weak_eff)
        pulse_weak = relu(weak_eff - 1000.0 * seen_weak)

        probe = relu(clock - thr_time)
        force = relu(probe - 1000.0 * seen_big - 1000.0 * seen_weak)
        mode_cw_new = relu(mode_cw + gate_gain * pulse_weak + gate_gain * force)

        # Update state for next step
        x_Ed = x_E
        x_E = energy
        seen_big = seen_big_new
        seen_weak = seen_weak_new
        mode_cw = mode_cw_new
        clock = relu(clock + clock_step)

        # Move after warmup
        if t > warmup:
            prev_pos = np.asarray(bot.position, dtype=float)
            dtheta = max(-rad5, min(rad5, O))
            bot.direction += dtheta
            T = np.array([np.cos(bot.direction), np.sin(bot.direction)], dtype=float)
            bot.position, bot.hit = bot.move_to(np.asarray(bot.position, dtype=float) + T * bot.speed, env)
            distance += float(np.linalg.norm(prev_pos - np.asarray(bot.position, dtype=float)))
            env.update(bot)

            prox = np.array(
                [proximity(bot.position, bot.direction + off) for off in offsets],
                dtype=float,
            )

    return distance


def primitives_search_player():
    rng = np.random.default_rng(12345)
    start = time.process_time()

    k = int(RAY_IDX.size)

    # Baseline: reproduce the strong CCW/CW wall-follow with only edge rays.
    thr0 = np.full(k, 0.99, dtype=float)
    thr0[0] = 0.88
    thr0[-1] = 0.79

    w_ccw0 = np.zeros(k, dtype=float)
    w_ccw0[0] = -50.0
    w_ccw0[-1] = 100.0

    w_cw0 = np.zeros(k, dtype=float)
    w_cw0[0] = -100.0
    w_cw0[-1] = 50.0

    mean = np.concatenate([thr0, w_ccw0, w_cw0])
    std = np.concatenate([
        np.full(k, 0.06, dtype=float),      # thresholds
        np.full(k, 40.0, dtype=float),      # CCW weights
        np.full(k, 40.0, dtype=float),      # CW weights
    ])

    baseline_params = mean.copy()
    baseline_model = build_model(baseline_params)
    best_params = baseline_params.copy()
    best_model = baseline_model

    # Yield baseline immediately so train() always has a result.
    yield best_model

    seed_pool = [101, 202, 303, 404, 505, 606, 707, 808]
    val_seeds = [111, 222, 333, 444]
    accept_margin = 0.05

    def score(params, seeds):
        return float(np.mean([rollout_distance(params, s) for s in seeds]))

    baseline_val = score(best_params, val_seeds)
    baseline_train = score(best_params, seed_pool[:2])
    best_val = baseline_val
    best_train = baseline_train

    pop = 18
    elite = 6
    train_k = 2

    while True:
        if time.process_time() - start > 94.0:
            break

        gen = int((time.process_time() - start) // 1.0)
        train_seeds = [seed_pool[(gen * train_k + i) % len(seed_pool)] for i in range(train_k)]

        candidates = rng.normal(mean, std, size=(pop, mean.size))

        # Clip candidates into reasonable bounds
        candidates[:, :k] = np.clip(candidates[:, :k], 0.2, 0.99)
        candidates[:, k:] = np.clip(candidates[:, k:], -200.0, 200.0)

        scores_list = []
        used = 0
        for cand in candidates:
            if time.process_time() - start > 94.0:
                break
            scores_list.append(score(cand, train_seeds))
            used += 1

        if used < elite:
            break

        candidates = candidates[:used]
        scores = np.array(scores_list, dtype=float)
        top = int(np.argmax(scores))

        # Validate only the current best candidate from this generation.
        if scores[top] > best_train + accept_margin:
            if time.process_time() - start <= 94.0:
                val_score = score(candidates[top], val_seeds)
            else:
                val_score = -np.inf
            if val_score > best_val + accept_margin:
                best_train = float(scores[top])
                best_val = float(val_score)
                best_params = candidates[top].copy()
                best_model = build_model(best_params)
                yield best_model

        elite_idx = np.argsort(scores)[-elite:]
        elite_params = candidates[elite_idx]
        mean = elite_params.mean(axis=0)
        std = 0.85 * std + 0.15 * elite_params.std(axis=0)

        # Keep some exploration alive
        std[:k] = np.maximum(std[:k], 0.015)
        std[k:] = np.maximum(std[k:], 10.0)

        yield best_model

    if best_val < baseline_val + accept_margin:
        best_model = baseline_model
    yield best_model
    return best_model


if __name__ == "__main__":
    seed = 12345
    np.random.seed(seed)

    print("Starting training for 100 seconds (user time)")
    from challenge_2 import train
    model = train(primitives_search_player, timeout=100)

    score, std = evaluate(model, Bot, Environment, runs=10, debug=False, seed=seed)
    print(f"Final score: {score:.2f} ± {std:.2f}")
