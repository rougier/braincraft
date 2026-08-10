# Bio Player — Environment 3

## 1. Overview

`env3_player_bio.py` is a pointwise-activation Echo State Network
controller for Environment 3:

```text
X(t+1) = f(Win @ I(t) + W @ X(t))
O(t+1) = Wout @ g(X(t+1))        (g = identity)
```

Every hidden activation is a scalar function of its own preactivation;
all cross-neuron logic lives in the connectivity matrices. The model
is produced by a single `yield` in `bio_player()`, so the matrices are
fixed at build time (no iterative training).

Env3 exposes colour (the sources are coloured), but this controller
does not read colour or energy. The bot runs around the outer
corridor with a reflex wall-follower and picks up whatever source it
happens to cross. Only 7 of the 1000 hidden slots receive any
incoming weight; the rest are dead.

## 2. Network shape

| Parameter     | Value                    |
| ------------- | ------------------------ |
| `n`           | `1000`                   |
| `p`           | `64` (camera rays)       |
| `n_inputs`    | `2*p + 3 = 131`          |
| `warmup`      | `0`                      |
| `leak` (λ)    | `1.0`                    |
| `g`           | identity                 |
| Actuator clip | `step_a = 5°`            |

Module constants: `front_gain_mag = 20°`, `step_a = 5°`.

## 3. Activations

| Name        | Formula                     | Used by                                |
| ----------- | --------------------------- | -------------------------------------- |
| `relu_tanh` | `max(0, tanh(z))`           | reflex channels (slots 0..4), front-block |
| `clip_a`    | `clip(z, -step_a, +step_a)` | `dtheta`                               |

## 4. Inputs and slot layout

```text
I(t) = [prox[0..63](t), colour[0..63](t), hit(t), energy(t), 1]
```

Taps used by the controller (colour and energy columns are unread):

```text
L_idx          = 20      (left reflex proximity tap)
R_idx          = 43      (right reflex proximity tap)
left_side_idx  = 11      (left safety tap)
right_side_idx = 52      (right safety tap)
C1_idx, C2_idx = 31, 32  (centre-front proximity taps)
hit_idx        = 128     (= 2*p)
bias_idx       = 130     (= 2*p + 2)
```

Hidden slots (`n = 1000`; slots `7..999` are dead):

| Slot | Name          | Activation  | Role                              |
| ---- | ------------- | ----------- | --------------------------------- |
| 0    | `hit_feat`    | `relu_tanh` | hit reflex                        |
| 1    | `prox_left`   | `relu_tanh` | left proximity reflex             |
| 2    | `prox_right`  | `relu_tanh` | right proximity reflex            |
| 3    | `safe_left`   | `relu_tanh` | left safety feature               |
| 4    | `safe_right`  | `relu_tanh` | right safety feature              |
| 5    | `dtheta`      | `clip_a`    | one-step-lagged steering command  |
| 6    | `front_block` | `relu_tanh` | unsigned front-block detector     |

## 5. Circuits

### 5.1 Reflex features and readout

Five feed-forward proximity/hit detectors:

```text
hit_feat(t+1)   = relu_tanh(hit(t))
prox_left(t+1)  = relu_tanh(prox[L_idx](t))
prox_right(t+1) = relu_tanh(prox[R_idx](t))
safe_left(t+1)  = relu_tanh(-prox[left_side_idx](t)  + 0.75)
safe_right(t+1) = relu_tanh(-prox[right_side_idx](t) + 0.75)
```

Steering readout:

```text
O(t+1) = hit_turn          * hit_feat(t+1)
       + heading_gain      * prox_left(t+1)
       - heading_gain      * prox_right(t+1)
       + safety_gain_left  * safe_left(t+1)
       + safety_gain_right * safe_right(t+1)
       + front_gain_mag    * front_block(t+1)
```

with

```text
hit_turn          = -10° / tanh(1)
heading_gain      = -40°
safety_gain_left  = -20°
safety_gain_right = +20°
front_gain_mag    = +20°
```

`dtheta` holds the clipped one-step-lagged command,
`dtheta(t+1) = clip(O(t), ±step_a)`, implemented by mirroring the
`Wout` row into `W[dtheta, :]`.

### 5.2 Front block

Unsigned sum of the two centre proximity taps:

```text
front_block(t+1) = relu_tanh(prox[C1_idx](t) + prox[C2_idx](t) - 1.4)
```

A positive reading turns the bot by `+20°` (CCW) — a fixed-direction
escape that keeps the bot on the outer corridor.

## 6. Nonzero readout weights

```text
Wout[hit_feat]    = -10° / tanh(1)
Wout[prox_left]   = -40°
Wout[prox_right]  = +40°
Wout[safe_left]   = -20°
Wout[safe_right]  = +20°
Wout[front_block] = +20°
```

Six nonzero entries total. The same row is mirrored into
`W[dtheta, :]`.

## 7. Verification

```bash
python braincraft/env3_player_bio.py
```

Runs `train(bio_player, timeout=100)` then
`evaluate(model, Bot, Environment, debug=False, seed=12345)` over 10
episodes:

```text
Final score (distance): 14.40 +/- 0.49
```

500-seed sweep (`validate_env3_player_bio.py`, seeds 0..499):
across-seed mean `14.50 ± 0.17`, min `14.00`, `0/500` seeds below
`13.50`.
