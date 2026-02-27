



from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from bot import Bot


def identity(x):
    return x


@dataclass
class _PolicyConfig:
    n_neurons: int = 1000
    camera_resolution: int = 64
    side_right_idx: int = 52
    side_left_idx: int = 11


    energy_smooth_leak: float = 0.05


    gamma: float = 0.995
    lr: float = 0.01
    action_sigma: float = 0.02
    grad_clip: float = 0.5


    hit_penalty_coeff: float = 0.001


    max_steps: int = 2500
    rng_seed: int = 12345
    max_seed_probe: int = 200_000


def _build_model_from_wout(
    wout: np.ndarray,
    *,
    cfg: _PolicyConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, callable, callable]:

    n = cfg.n_neurons
    p = cfg.camera_resolution
    n_inp = 2 * p + 3

    if wout.shape != (6,):
        raise ValueError("wout must have shape (6,)")

    hit_idx = 2 * p
    energy_idx = 2 * p + 1
    bias_idx = 2 * p + 2
    side_right_idx = int(np.clip(int(cfg.side_right_idx), 0, p - 1))
    side_left_idx = int(np.clip(int(cfg.side_left_idx), 0, p - 1))

    Win = np.zeros((n, n_inp), dtype=np.float64)
    W = np.zeros((n, n), dtype=np.float64)
    Wout = np.zeros((1, n), dtype=np.float64)


    Win[0, hit_idx] = 1.0
    Win[1, bias_idx] = 1.0
    Win[2, side_right_idx] = 1.0
    Win[3, side_left_idx] = 1.0
    Win[4, energy_idx] = 1.0
    Win[5, energy_idx] = 1.0

    Wout[0, 0:6] = wout

    warmup = 0

    leak = np.ones((n, 1), dtype=np.float64)
    leak[5, 0] = float(np.clip(cfg.energy_smooth_leak, 1e-4, 1.0))

    f = identity
    g = identity
    return Win, W, Wout, warmup, leak, f, g


def _rollout_episode(
    *,
    env,
    cfg: _PolicyConfig,
    wout: np.ndarray,
    rng: np.random.Generator,
    training: bool,
    deadline: float | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:

    max_turn = float(np.radians(5.0))

    bot = Bot()
    cam = bot.camera


    D = 0.25
    Wp = 2 * D * np.tan(np.radians(cam.fov) / 2)

    def _offset(idx: int) -> float:
        x = (Wp / 2) * (1.0 - 2.0 * float(idx) / float(max(1, cam.resolution - 1)))
        return float(np.arctan2(x, D))

    off_r = _offset(int(cfg.side_right_idx))
    off_l = _offset(int(cfg.side_left_idx))

    def _proximity(position, direction: float, off: float) -> float:
        end, _cell, _face, _steps = cam.raycast(position, direction + off, env.world)
        if end is None:
            return 0.0
        d = float(np.linalg.norm(np.asarray(position, dtype=np.float64) - end))
        return 1.0 - d



    x = np.zeros(6, dtype=np.float64)


    energy_prev = float(bot.energy)


    feats = []
    mus = []
    acts = []
    rews = []

    distance = 0.0

    if deadline is not None:
        deadline = float(deadline)

    for _t in range(cfg.max_steps):
        if deadline is not None and (_t & 63) == 0 and time.process_time() >= deadline:
            break
        if bot.energy <= 0:
            break

        hit = float(bot.hit)
        bias = 1.0


        sideR = _proximity(bot.position, float(bot.direction), off_r)
        sideL = _proximity(bot.position, float(bot.direction), off_l)
        energy = float(bot.energy)


        x[0] = hit
        x[1] = bias
        x[2] = sideR
        x[3] = sideL
        x[4] = energy
        x[5] = (1.0 - cfg.energy_smooth_leak) * x[5] + cfg.energy_smooth_leak * energy

        mu = float(np.dot(wout, x))
        mu = float(np.clip(mu, -max_turn, max_turn))

        if training:
            a = float(mu + cfg.action_sigma * rng.standard_normal())
            a = float(np.clip(a, -max_turn, max_turn))
        else:
            a = mu


        if _t <= 0:
            continue


        p = np.asarray(bot.position, dtype=np.float64)
        bot.direction += float(np.clip(a, -max_turn, max_turn))
        T = np.array([np.cos(bot.direction), np.sin(bot.direction)], dtype=np.float64)
        bot.position, bot.hit = bot.move_to(p + T * bot.speed, env)
        env.update(bot)
        step_dist = float(np.linalg.norm(p - np.asarray(bot.position, dtype=np.float64)))
        distance += step_dist


        energy_now = float(bot.energy)
        delta_e = energy_now - energy_prev
        energy_prev = energy_now


        refill = delta_e + bot.energy_move + bot.energy_hit * float(bool(bot.hit))
        if refill < 0:
            refill = 0.0


        reward = step_dist + 10.0 * refill - cfg.hit_penalty_coeff * float(bool(bot.hit))

        if training:
            feats.append(x.copy())
            mus.append(mu)
            acts.append(a)
            rews.append(reward)

    if not training:
        return distance, np.zeros(6), np.ones(6)



    if len(rews) == 0:
        return distance, np.zeros_like(wout), np.ones(6, dtype=np.float64)


    rews = np.asarray(rews, dtype=np.float64)
    feats = np.asarray(feats, dtype=np.float64)
    mus = np.asarray(mus, dtype=np.float64)
    acts = np.asarray(acts, dtype=np.float64)

    returns = np.zeros_like(rews)
    running = 0.0
    for i in range(rews.size - 1, -1, -1):
        running = rews[i] + cfg.gamma * running
        returns[i] = running


    adv = returns - float(np.mean(returns))


    sigma2 = cfg.action_sigma * cfg.action_sigma

    grad = np.einsum("t,ti->i", (acts - mus) / sigma2 * adv, feats)
    grad /= max(1, feats.shape[0])

    feat_rms = np.sqrt(np.mean(feats * feats, axis=0) + 1e-12)
    episode_score = float(distance)
    return episode_score, grad, feat_rms


def energy_reinforce_player(timeout: float = 100.0) -> Iterable[Tuple]:

    cfg = _PolicyConfig()
    start = time.process_time()
    deadline = start + float(timeout) - 0.5



    hit_turn = 5.0
    wall_gain = 0.40
    wall_target = 0.65


    w = np.array(
        [
            hit_turn,
            -wall_gain * wall_target,
            wall_gain,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(int(cfg.rng_seed))


    yield _build_model_from_wout(w, cfg=cfg)


    best_w = w.copy()
    best_score = -float("inf")



    def _seed_good_id(seed: int) -> int:
        np.random.seed(int(seed))
        from environment_3 import Environment

        env = Environment()
        refills = {s.identity: float(s.refill) for s in env.sources}
        if not refills:
            return 0
        return int(max(refills, key=refills.get))

    from environment_3 import Environment

    eval_seeds = []
    source_ids = sorted({int(s.identity) for s in Environment().sources})
    if not source_ids:
        source_ids = [-1, -2]

    want_each = 4
    have = {sid: 0 for sid in source_ids}
    probe = 0
    while any(have[sid] < want_each for sid in source_ids) and probe < cfg.max_seed_probe:
        gid = _seed_good_id(probe)
        if gid in have and have[gid] < want_each:
            eval_seeds.append(probe)
            have[gid] += 1
        probe += 1


    if not eval_seeds:
        eval_seeds = [0, 1, 2, 3]

    def _eval_mean_distance(wout_vec: np.ndarray) -> float:
        scores = []
        for seed in eval_seeds:
            if time.process_time() >= deadline:
                break
            np.random.seed(int(seed))
            from environment_3 import Environment

            env = Environment()
            dist, _grad, _rms = _rollout_episode(
                env=env,
                cfg=cfg,
                wout=wout_vec,
                rng=rng,
                training=False,
                deadline=deadline,
            )
            scores.append(dist)
        return float(np.mean(scores)) if scores else 0.0


    best_score = _eval_mean_distance(best_w)

    lr = cfg.lr
    ema_feat_rms = np.ones_like(w)
    it = 0

    while True:
        if time.process_time() >= deadline:
            break
        it += 1

        from environment_3 import Environment

        env = Environment()

        ep_score, grad, feat_rms = _rollout_episode(
            env=env,
            cfg=cfg,
            wout=w,
            rng=rng,
            training=True,
            deadline=deadline,
        )


        ema_feat_rms = 0.95 * ema_feat_rms + 0.05 * feat_rms
        grad = grad / ema_feat_rms


        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > cfg.grad_clip:
            grad *= cfg.grad_clip / (grad_norm + 1e-12)

        w = w + lr * grad


        w = np.clip(w, -10.0, 10.0)



        if it % 10 == 0 and time.process_time() < deadline - 0.5:
            score = _eval_mean_distance(w)
            if score > best_score:
                best_score = score
                best_w = w.copy()


        yield _build_model_from_wout(best_w, cfg=cfg)


    yield _build_model_from_wout(best_w, cfg=cfg)



if __name__ == "__main__":
    from environment_3 import Environment
    from challenge_3 import train, evaluate

    seed = 12345
    np.random.seed(seed)

    print("Starting training for 100 seconds (user time)")
    model = train(energy_reinforce_player, timeout=100)

    print("Evaluating final model...")
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")
