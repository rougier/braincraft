

import time
import numpy as np

from bot import Bot
from environment_3 import Environment
from challenge_3 import evaluate


def relu(x):
    return np.clip(x, a_min=0.0, a_max=None)


def identity(x):
    return x


RAY_IDX = np.array([0, 16, 32, 48, 63], dtype=int)


def _clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


def build_model(params):
    k = int(RAY_IDX.size)
    thr = np.array(params[:k], dtype=float)
    w_ccw = np.array(params[k:2 * k], dtype=float)
    w_cw = np.array(params[2 * k:3 * k], dtype=float)
    thr_urgency = float(params[3 * k]) if len(params) > 3 * k else 0.5
    relief = float(params[3 * k + 1]) if len(params) > 3 * k + 1 else 1.0

    thr = np.clip(thr, 0.2, 0.99)
    w_ccw = np.clip(w_ccw, -200.0, 200.0)
    w_cw = np.clip(w_cw, -200.0, 200.0)
    thr_urgency = float(np.clip(thr_urgency, 0.1, 1.5))
    relief = float(np.clip(relief, 0.0, 5.0))

    bot = Bot()
    n = 1000
    n_cam = bot.camera.resolution
    n_inp = 2 * n_cam + 3

    i_energy = 2 * n_cam + 1
    i_bias = 2 * n_cam + 2


    t_bp = 0.003
    thr_any = 1e-5
    thr_big = 1e-3
    gate_gain = 4167.0


    i_E = 100
    i_Ed = 101
    i_dE = 102
    i_drop = 103

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
    i_urgency = 121
    i_force = 123
    i_mode_cw_over = 124


    i_ccw0 = 1
    i_cw0 = i_ccw0 + k

    W_in = np.zeros((n, n_inp))
    W = np.zeros((n, n))
    W_out = np.zeros((1, n))



    W_in[i_E, i_energy] = 1.0
    W[i_Ed, i_E] = 1.0
    W[i_dE, i_E] = 1.0
    W[i_dE, i_Ed] = -1.0


    W[i_drop, i_Ed] = 1.0
    W[i_drop, i_E] = -1.0

    W[i_f1, i_dE] = 1.0
    W_in[i_f1, i_bias] = -t_bp
    W[i_f2, i_dE] = 1.0
    W[i_bp, i_f1] = -2.0
    W[i_bp, i_f2] = 1.0


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


    W[i_urgency, i_urgency] = 1.0
    W[i_urgency, i_drop] = 1.0
    W[i_urgency, i_any] = -relief

    W[i_force, i_urgency] = 1.0
    W_in[i_force, i_bias] = -thr_urgency
    W[i_force, i_seen_big] = -1000.0
    W[i_force, i_seen_weak] = -1000.0


    W[i_mode_cw, i_mode_cw] = 1.0
    W[i_mode_cw, i_pulse_weak] = gate_gain
    W[i_mode_cw, i_force] = gate_gain
    W[i_mode_cw, i_mode_cw_over] = -1.0
    W[i_mode_cw_over, i_mode_cw] = 1.0
    W_in[i_mode_cw_over, i_bias] = -1.0



    for j, idx in enumerate(RAY_IDX):
        u_ccw = i_ccw0 + j
        u_cw = i_cw0 + j


        W_in[u_ccw, int(idx)] = 1.0
        W_in[u_ccw, i_bias] = -float(thr[j])
        W[u_ccw, i_mode_cw] = -1.0
        W_out[0, u_ccw] = float(w_ccw[j])


        W_in[u_cw, int(idx)] = 1.0
        W_in[u_cw, i_bias] = -(float(thr[j]) + 1.0)
        W[u_cw, i_mode_cw] = 1.0
        W_out[0, u_cw] = float(w_cw[j])

    warmup = 1
    leak = 1.0
    f = relu
    g = identity
    return (W_in, W, W_out, warmup, leak, f, g)


def rollout_distance(params, seed, max_steps=4000, deadline=None):
    np.random.seed(seed)
    bot = Bot()
    env = Environment()
    cam = bot.camera

    k = int(RAY_IDX.size)
    thr = np.clip(np.array(params[:k], dtype=float), 0.2, 0.99)
    w_ccw = np.clip(np.array(params[k:2 * k], dtype=float), -200.0, 200.0)
    w_cw = np.clip(np.array(params[2 * k:3 * k], dtype=float), -200.0, 200.0)
    thr_urgency = float(params[3 * k]) if len(params) > 3 * k else 0.5
    relief = float(params[3 * k + 1]) if len(params) > 3 * k + 1 else 1.0
    thr_urgency = float(np.clip(thr_urgency, 0.1, 1.5))
    relief = float(np.clip(relief, 0.0, 5.0))


    t_bp = 0.003
    thr_any = 1e-5
    thr_big = 1e-3
    gate_gain = 4167.0


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


    x_E = 0.0
    x_Ed = 0.0
    x_dE = 0.0
    x_drop = 0.0
    x_f1 = 0.0
    x_f2 = 0.0
    x_bp = 0.0
    x_any = 0.0
    x_big = 0.0
    x_weak = 0.0
    x_weak_eff = 0.0
    seen_big = 0.0
    seen_weak = 0.0
    x_pulse_weak = 0.0
    mode_cw = 0.0
    mode_cw_over = 0.0
    urgency = 0.0
    x_force = 0.0

    distance = 0.0
    warmup = 1
    rad5 = float(np.radians(5))

    for t in range(max_steps):
        if deadline is not None and (t & 63) == 0 and time.process_time() >= float(deadline):
            break
        if bot.energy <= 0:
            break

        energy = float(bot.energy)


        m = mode_cw
        ccw_feat = relu(prox - thr - m)
        cw_feat = relu(prox + m - (thr + 1.0))
        O = float(np.dot(w_ccw, ccw_feat) + np.dot(w_cw, cw_feat))


        dE_new = relu(x_E - x_Ed)
        drop_new = relu(x_Ed - x_E)
        f1_new = relu(x_dE - t_bp)
        f2_new = relu(x_dE)
        bp_new = relu(-2.0 * x_f1 + x_f2)
        any_new = relu(x_bp - thr_any)
        big_new = relu(x_bp - thr_big)
        weak_new = relu(x_any - 1000.0 * x_big)

        seen_big_new = relu(seen_big + x_big)
        weak_eff_new = relu(x_weak - 1000.0 * seen_big)
        seen_weak_new = relu(seen_weak + x_weak_eff)
        pulse_weak_new = relu(x_weak_eff - 1000.0 * seen_weak)

        urgency_new = relu(urgency + x_drop - relief * x_any)
        force_new = relu(urgency - thr_urgency - 1000.0 * seen_big - 1000.0 * seen_weak)
        mode_over_new = relu(mode_cw - 1.0)
        mode_cw_new = relu(mode_cw + gate_gain * x_pulse_weak + gate_gain * x_force - mode_cw_over)


        x_Ed = x_E
        x_E = energy
        x_dE = dE_new
        x_drop = drop_new
        x_f1 = f1_new
        x_f2 = f2_new
        x_bp = bp_new
        x_any = any_new
        x_big = big_new
        x_weak = weak_new
        x_weak_eff = weak_eff_new
        seen_big = seen_big_new
        seen_weak = seen_weak_new
        x_pulse_weak = pulse_weak_new
        mode_cw = mode_cw_new
        mode_cw_over = mode_over_new
        urgency = urgency_new
        x_force = force_new


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


def primitives_search_valuation_player():
    rng = np.random.default_rng(12345)
    start = time.process_time()
    deadline = start + 94.0

    k = int(RAY_IDX.size)


    thr0 = np.full(k, 0.99, dtype=float)
    thr0[0] = 0.88
    thr0[-1] = 0.79

    w_ccw0 = np.zeros(k, dtype=float)
    w_ccw0[0] = -50.0
    w_ccw0[-1] = 100.0

    w_cw0 = np.zeros(k, dtype=float)
    w_cw0[0] = -100.0
    w_cw0[-1] = 50.0


    thr_urg0 = 0.50
    relief0 = 1.00

    mean = np.concatenate([thr0, w_ccw0, w_cw0, np.array([thr_urg0, relief0], dtype=float)])
    std = np.concatenate([
        np.full(k, 0.06, dtype=float),
        np.full(k, 40.0, dtype=float),
        np.full(k, 40.0, dtype=float),
        np.array([0.20, 0.75], dtype=float)
    ])

    baseline_params = mean.copy()
    baseline_model = build_model(baseline_params)
    best_params = baseline_params.copy()
    best_model = baseline_model


    yield best_model

    _gid_cache = {}

    def _best_source_id(seed: int) -> int:
        s = int(seed)
        if s in _gid_cache:
            return _gid_cache[s]
        np.random.seed(s)
        env = Environment()
        refills = {int(s.identity): float(s.refill) for s in env.sources}
        gid = int(max(refills, key=refills.get)) if refills else 0
        _gid_cache[s] = gid
        return gid

    source_ids = sorted({int(s.identity) for s in Environment().sources})
    if not source_ids:
        source_ids = [-1, -2]

    def _balanced_seeds(*, want_each: int, start_seed: int, max_probe: int = 200_000) -> list[int]:
        seeds: list[int] = []
        have = {sid: 0 for sid in source_ids}
        s = int(start_seed)
        probe = 0
        while any(have[sid] < want_each for sid in source_ids):
            if probe >= max_probe:
                break
            gid = _best_source_id(s)
            if gid in have and have[gid] < want_each:
                seeds.append(s)
                have[gid] += 1
            s += 1
            probe += 1
        if not seeds:
            return [start_seed, start_seed + 1, start_seed + 2, start_seed + 3]
        return seeds

    seed_pool = _balanced_seeds(want_each=4, start_seed=0)
    val_seeds = _balanced_seeds(want_each=2, start_seed=10_000)
    pool_by_gid = {sid: [s for s in seed_pool if _best_source_id(s) == sid] for sid in source_ids}
    active_source_ids = [sid for sid in source_ids if pool_by_gid[sid]]
    if not active_source_ids:
        fallback_sid = source_ids[0] if source_ids else 0
        active_source_ids = [fallback_sid]
        pool_by_gid = {fallback_sid: seed_pool[:] if seed_pool else [0]}
    accept_margin = 0.05

    def score(params, seeds):
        scores = []
        for s in seeds:
            if time.process_time() >= deadline:
                break
            scores.append(rollout_distance(params, s, deadline=deadline))
        return float(np.mean(scores)) if scores else -np.inf

    baseline_val = score(best_params, val_seeds)
    baseline_train = score(best_params, [pool_by_gid[sid][0] for sid in active_source_ids])
    best_val = baseline_val
    best_train = baseline_train

    pop = 18
    elite = 6
    while True:
        if time.process_time() >= deadline:
            break

        gen = int((time.process_time() - start) // 1.0)
        train_seeds = [pool_by_gid[sid][gen % len(pool_by_gid[sid])] for sid in active_source_ids]

        candidates = rng.normal(mean, std, size=(pop, mean.size))


        candidates[:, :k] = np.clip(candidates[:, :k], 0.2, 0.99)
        candidates[:, k:] = np.clip(candidates[:, k:], -200.0, 200.0)

        candidates[:, 3 * k] = np.clip(candidates[:, 3 * k], 0.1, 1.5)
        candidates[:, 3 * k + 1] = np.clip(candidates[:, 3 * k + 1], 0.0, 5.0)

        scores_list = []
        used = 0
        for cand in candidates:
            if time.process_time() >= deadline:
                break
            scores_list.append(score(cand, train_seeds))
            used += 1

        if used < elite:
            break

        candidates = candidates[:used]
        scores = np.array(scores_list, dtype=float)
        top = int(np.argmax(scores))


        if scores[top] > best_train + accept_margin:
            if time.process_time() < deadline:
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


        std[:k] = np.maximum(std[:k], 0.015)
        std[k:3 * k] = np.maximum(std[k:3 * k], 10.0)
        std[3 * k] = max(std[3 * k], 0.05)
        std[3 * k + 1] = max(std[3 * k + 1], 0.20)

        yield best_model

    if best_val < baseline_val + accept_margin:
        best_model = baseline_model
    yield best_model
    return best_model


if __name__ == "__main__":
    seed = 12345
    np.random.seed(seed)

    print("Starting training for 100 seconds (user time)")
    from challenge_3 import train
    model = train(primitives_search_valuation_player, timeout=100)

    score, std = evaluate(model, Bot, Environment, runs=10, debug=False, seed=seed)
    print(f"Final score: {score:.2f} ± {std:.2f}")
