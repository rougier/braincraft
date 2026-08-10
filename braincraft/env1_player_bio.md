# Bio Player — Environment 1

## 1. Overview

`env1_player_bio.py` is a pointwise-activation Echo State Network
controller for Environment 1. Every hidden activation is a scalar
function of its own preactivation; all cross-neuron logic lives in the
connectivity matrices:

```text
X(t+1) = f(Win @ I(t) + W @ X(t))
O(t+1) = Wout @ g(X(t+1))        (g = identity)
```

The model is produced by a single `yield` in `bio_player()`, so the
matrices are fixed at build time (no iterative training).

The controller combines four behaviours:

- a **reflex wall-follower** (proximity, safety, hit),
- an **initial-heading correction** that erases the ±5° start-direction
  perturbation and keeps `pos_x` / `pos_y` integration faithful,
- a **rising-edge energy-reward detector** that arms the shortcut, and
- a **pose-gated corridor shortcut** that fires when the reward has been
  seen and the bot crosses a clear horizontal corridor.

The env2 bio player extends this design with a colour-evidence circuit
and a signed front-block; see `env2_player_bio.md` for that variant.

## 2. Network shape

| Parameter     | Value                         |
| ------------- | ----------------------------- |
| `n`           | `1000`                        |
| `p`           | `64` (camera rays)            |
| `n_inputs`    | `p + 3 = 67`                  |
| `warmup`      | `0`                           |
| `leak` (λ)    | `1.0`                         |
| `g`           | identity                      |
| Actuator clip | `step_a = 5°`                 |
| Bot speed     | `0.01` (`bot.speed`)          |

Module constants:

```text
shortcut_turn  = -2.0          turn_steps     = 18
near_c_thr     = 0.05          approach_steps = 50
drift_offset   = 0.175         sc_total       = 68
k_sharp        = 50.0          ncr_gain       = 2.5
front_gain_mag = 20°           near_cr_gain   = 2.5
step_a         = 5°            seed_window_k  = 6
cal_gain       = 1 / 0.173
```

## 3. Activation library

Each slot has a fixed scalar activation applied pointwise to its
preactivation `z`:

| Name        | Formula                     | Used by                                                                                                  |
| ----------- | --------------------------- | -------------------------------------------------------------------------------------------------------- |
| `relu_tanh` | `max(0, tanh(z))`           | default — threshold / latch / AND-OR gates                                                               |
| `identity`  | `z`                         | `dir_accum`, `pos_x`, `pos_y`, `head_corr`, `shortcut_steer`, `init_impulse`, `step_counter`             |
| `relu`      | `max(0, z)`                 | `energy_ramp`, `sc_countdown`                                                                            |
| `clip_a`    | `clip(z, -step_a, +step_a)` | `dtheta`                                                                                                 |
| `sin`       | `sin(z)`                    | `sin_n`, `cos_n`                                                                                         |
| `bump`      | `max(0, 1 - 4 z^2)`         | `near_e`, `near_w`                                                                                       |

## 4. Inputs and slot layout

```text
I(t) = [prox[0..63](t), hit(t), energy(t), 1]
```

Input taps used by the controller:

```text
L_idx          = 20     (left reflex proximity tap)
R_idx          = 43     (right reflex proximity tap)
left_side_idx  = 11     (left safety tap)
right_side_idx = 52     (right safety tap)
C1_idx, C2_idx = 31, 32 (centre-front proximity taps)
hit_idx        = 64
energy_idx     = 65
bias_idx       = 66
```

Hidden slots (`n = 1000`; slots `40..999` receive no incoming weights):

| Slot | Name              | Activation  | Role                                               |
| ---- | ----------------- | ----------- | -------------------------------------------------- |
| 0    | `hit_feat`        | `relu_tanh` | hit reflex                                         |
| 1    | `prox_left`       | `relu_tanh` | left proximity reflex                              |
| 2    | `prox_right`     | `relu_tanh` | right proximity reflex                             |
| 3    | `safe_left`       | `relu_tanh` | left safety feature                                |
| 4    | `safe_right`      | `relu_tanh` | right safety feature                               |
| 5    | `dtheta`          | `clip_a`    | one-step-lagged steering command                   |
| 6    | `dir_accum`       | `identity`  | integrated heading                                 |
| 7    | `pos_x`           | `identity`  | integrated x position                              |
| 8    | `pos_y`           | `identity`  | integrated y position                              |
| 9    | `head_corr`       | `identity`  | latched initial-heading correction                 |
| 10   | `seeded_flag`     | `relu_tanh` | seed-window latch (also arms the reward pulse)     |
| 11   | `step_counter`    | `identity`  | step index (drives `seeded_flag` timing)           |
| 12   | `seed_pos`        | `relu_tanh` | positive initial-correction pulse                  |
| 13   | `seed_neg`        | `relu_tanh` | negative initial-correction pulse                  |
| 14   | `energy_ramp`     | `relu`      | previous-step energy                               |
| 15   | `reward_pulse`    | `relu_tanh` | energy rising-edge detector                        |
| 16   | `reward_latch`    | `relu_tanh` | latched reward-seen signal                         |
| 17   | `sc_countdown`    | `relu`      | shortcut phase countdown                           |
| 18   | `shortcut_steer`  | `identity`  | shortcut steering actuator                         |
| 19   | `init_impulse`    | `identity`  | initial-correction steering actuator               |
| 20   | `sin_n`           | `sin`       | `sin(phi)`                                         |
| 21   | `cos_n`           | `sin`       | `cos(phi)` (via `sin(phi + π/2)`)                  |
| 22   | `sin_pos`         | `relu_tanh` | `sin(phi) > 0` sharp detector                      |
| 23   | `sin_neg`         | `relu_tanh` | `sin(phi) < 0` sharp detector                      |
| 24   | `y_pos`           | `relu_tanh` | `pos_y > 0` sharp detector                         |
| 25   | `y_neg`           | `relu_tanh` | `pos_y < 0` sharp detector                         |
| 26   | `near_e`          | `bump`      | east-shifted corridor bump                         |
| 27   | `near_w`          | `bump`      | west-shifted corridor bump                         |
| 28   | `near_cr_e`       | `relu_tanh` | AND(`near_e`, heading east)                        |
| 29   | `near_cr_w`       | `relu_tanh` | AND(`near_w`, heading west)                        |
| 30   | `near_cr`         | `relu_tanh` | corridor predicate (OR of the two)                 |
| 31   | `trig_sc`         | `relu_tanh` | shortcut trigger pulse                             |
| 32   | `on_countdown`    | `relu_tanh` | `sc_countdown > 0.5`                               |
| 33   | `is_turn`         | `relu_tanh` | turn-phase gate                                    |
| 34   | `is_app`          | `relu_tanh` | approach-phase gate                                |
| 35   | `sy_pp`           | `relu_tanh` | AND(`sin_pos`, `y_pos`, `is_turn`)                 |
| 36   | `sy_pn`           | `relu_tanh` | AND(`sin_pos`, `y_neg`, `is_turn`)                 |
| 37   | `sy_np`           | `relu_tanh` | AND(`sin_neg`, `y_pos`, `is_turn`)                 |
| 38   | `sy_nn`           | `relu_tanh` | AND(`sin_neg`, `y_neg`, `is_turn`)                 |
| 39   | `front_block`     | `relu_tanh` | unsigned front-block detector                      |

## 5. Main circuits

### 5.1 Reflex features and readout

Reflex channels, suppressed during the shortcut approach phase:

```text
hit_feat(t+1)   = relu_tanh(hit(t)                          - k_sharp * is_app(t))
prox_left(t+1)  = relu_tanh(prox[L_idx](t)                  - k_sharp * is_app(t))
prox_right(t+1) = relu_tanh(prox[R_idx](t)                  - k_sharp * is_app(t))
safe_left(t+1)  = relu_tanh(-prox[left_side_idx](t)  + 0.75 - k_sharp * is_app(t))
safe_right(t+1) = relu_tanh(-prox[right_side_idx](t) + 0.75 - k_sharp * is_app(t))
```

Steering readout:

```text
O(t+1) = hit_turn          * hit_feat(t+1)
       + heading_gain      * prox_left(t+1)
       - heading_gain      * prox_right(t+1)
       + safety_gain_left  * safe_left(t+1)
       + safety_gain_right * safe_right(t+1)
       + front_gain_mag    * front_block(t+1)
       + shortcut_steer(t+1)
       + init_impulse(t+1)
```

with

```text
hit_turn          = -10° / tanh(1)
heading_gain      = -40°
safety_gain_left  = -20°
safety_gain_right = +20°
front_gain_mag    = +20°
```

The `dtheta` slot holds the clipped one-step-lagged command,
`dtheta(t+1) = clip(O(t), ±step_a)`, implemented by mirroring `Wout`
row 0 into `W[dtheta, :]`.

### 5.2 Heading, trig, and position

```text
dir_accum(t+1) = dir_accum(t) + dtheta(t)

phi(t)      = dir_accum(t) + head_corr(t) + dtheta(t)
sin_n(t+1)  = sin(phi(t))
cos_n(t+1)  = sin(phi(t) + π/2)       # = cos(phi(t))

pos_x(t+1) = pos_x(t) - speed * sin_n(t)
pos_y(t+1) = pos_y(t) + speed * cos_n(t)
```

`phi` is measured relative to north: `phi = 0` points north, `-π/2`
east, `+π/2` west. The `head_corr` term inside the trig neurons
compensates for the random ±5° start-direction perturbation (see §5.3)
so that `sin_n` / `cos_n` — and the `pos_x` / `pos_y` integrators that
read them — track the bot's true heading and position faithfully.

### 5.3 Initial-heading correction

The raw signed correction is

```text
current_corr(t) = (prox[R_idx](t) - prox[L_idx](t)) * cal_gain
```

The latch is built from `step_counter`, `seeded_flag`, `seed_pos`,
`seed_neg`, `head_corr`, and `init_impulse`. `step_counter` is an
identity-activation counter (`step_counter(t) = t`), and `seeded_flag`
is a sharp threshold against it:

```text
step_counter(t+1) = step_counter(t) + 1
seeded_flag(t+1)  = relu_tanh(k_sharp * (step_counter(t) - (seed_window_k - 1.5)))

seed_pos(t+1) = relu_tanh(-cal_gain*prox[L_idx](t) + cal_gain*prox[R_idx](t)
                          - 1000 * seeded_flag(t))
seed_neg(t+1) = relu_tanh( cal_gain*prox[L_idx](t) - cal_gain*prox[R_idx](t)
                          - 1000 * seeded_flag(t))

head_corr(t+1)    = head_corr(t) + seed_pos(t) - seed_neg(t)
init_impulse(t+1) = -seed_pos(t) + seed_neg(t)
```

With `seed_window_k = 6`, `seeded_flag(t) = 0` for `t = 0..5` and
saturates to `1` for `t ≥ 6`. Because `seed_pos(t+1)` reads
`seeded_flag(t)`, the seeds fire for six consecutive network steps
(`t = 1..6`). `head_corr` integrates each step's residual depth
asymmetry across the window while `init_impulse` steers against the
same seed — the net output contribution cancels, so the bot does not
physically turn while the controller's internal heading frame is being
aligned. The closed loop drives the residual heading error toward zero
across the six-step window.

### 5.4 Reward circuit

With `pulse_gain = 500`, `pulse_thr = 0.2`, `arm_gate = 1000`,
`latch_gain = 10`:

```text
energy_ramp(t+1) = relu(energy(t))

reward_pulse(t+1) = relu_tanh(
    pulse_gain * energy(t)
  - pulse_gain * energy_ramp(t)
  + arm_gate   * seeded_flag(t)
  - (arm_gate + pulse_thr)
)

reward_latch(t+1) = relu_tanh(latch_gain * reward_pulse(t)
                              + latch_gain * reward_latch(t))
```

`seeded_flag` doubles as the reward arm gate: the pulse is held off by
the `-(arm_gate + pulse_thr)` bias for the first `seed_window_k` steps
and, once the seed window closes, reduces to a sharp rising-edge
detector on `energy(t) - energy(t-1)`. The bot cannot reach any source
inside the short seed window, so arming only after it closes is safe.

### 5.5 Corridor tests and shortcut trigger

Two bump detectors on `pos_x`:

```text
near_e(t+1) = bump((pos_x(t) + drift_offset) / (2*near_c_thr))
near_w(t+1) = bump((pos_x(t) - drift_offset) / (2*near_c_thr))
```

`near_e` peaks near `pos_x = -drift_offset` (east-bound leg); `near_w`
is the mirror. Each is combined with a heading check. Because `phi` is
measured from north, `sin_n = sin(phi) ≈ -1` when heading east and
`≈ +1` when heading west:

```text
near_cr_e(t+1) = relu_tanh(ncr_gain * k_sharp * (near_e(t) - sin_n(t) - 1.5))
near_cr_w(t+1) = relu_tanh(ncr_gain * k_sharp * (near_w(t) + sin_n(t) - 1.5))
```

The `±0.5` margin accepts headings within `~±60°` of horizontal
(normal approaches have `|sin_n| > 0.95` at the trigger) while
rejecting perpendicular crossings of `pos_x = ±drift_offset` on later
laps. `ncr_gain = 2.5` sharpens the AND so near-threshold inputs don't
fire partially.

The corridor predicate is a sharpened OR:

```text
near_cr(t+1) = relu_tanh(near_cr_gain * k_sharp * (near_cr_e(t) + near_cr_w(t) - 0.5))
```

The trigger is a 2-way AND with two refractory terms:

```text
trig_sc(t+1) = relu_tanh(
    k_sharp * (reward_latch(t) + near_cr(t) - 1.5)
  - 10 * k_sharp * trig_sc(t)
  -      k_sharp * sc_countdown(t)
)
```

### 5.6 Shortcut countdown, phases, and steering

```text
sc_countdown(t+1) = relu(sc_countdown(t) - 1 + (sc_total + 1) * trig_sc(t))

on_countdown(t+1) = relu_tanh(k_sharp * (sc_countdown(t) - 0.5))
is_turn(t+1)      = relu_tanh(k_sharp * (sc_countdown(t) - (approach_steps + 0.5)))
is_app(t+1)       = relu_tanh(k_sharp * (on_countdown(t) - is_turn(t) - 0.5))
```

During the turn phase, four quadrant ANDs implement
`turn_toward = sign(sin(phi)) · sign(pos_y)`:

```text
sy_pp(t+1) = relu_tanh(k_sharp * (sin_pos(t) + y_pos(t) + is_turn(t) - 2.5))
sy_pn(t+1) = relu_tanh(k_sharp * (sin_pos(t) + y_neg(t) + is_turn(t) - 2.5))
sy_np(t+1) = relu_tanh(k_sharp * (sin_neg(t) + y_pos(t) + is_turn(t) - 2.5))
sy_nn(t+1) = relu_tanh(k_sharp * (sin_neg(t) + y_neg(t) + is_turn(t) - 2.5))

shortcut_steer(t+1) = |shortcut_turn| * (sy_pp(t) + sy_nn(t) - sy_pn(t) - sy_np(t))
```

### 5.7 Front block

An unsigned front-block from the two centre proximity taps:

```text
front_block(t+1) = relu_tanh(C1(t) + C2(t) - front_thr)
```

with `front_thr = 1.4`. A positive reading turns the bot by
`+front_gain_mag = +20°` (CCW) — a fixed-direction escape.

## 6. Nonzero readout weights

```text
Wout[hit_feat]        = hit_turn          = -10° / tanh(1)
Wout[prox_left]       = heading_gain      = -40°
Wout[prox_right]      = -heading_gain     = +40°
Wout[safe_left]       = safety_gain_left  = -20°
Wout[safe_right]      = safety_gain_right = +20°
Wout[front_block]     = +front_gain_mag   = +20°
Wout[shortcut_steer]  = +1
Wout[init_impulse]    = +1
```

All other `Wout` entries are zero (eight nonzero weights). The same
row is mirrored into `W[dtheta, :]` so that `dtheta(t+1) = clip(O(t),
±step_a)`.

## 7. Verification

```bash
python braincraft/env1_player_bio.py
```

Runs `train(bio_player, timeout=100)` followed by
`evaluate(model, Bot, Environment, debug=False, seed=12345)` over 10
episodes. Observed output:

```text
Final score (distance): 14.40 +/- 0.36
```