
<img src="./data/braincraft.png" width="100%">

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Introduction](#introduction)
- [Tasks](#tasks)
- [Methods](#methods)
- [Evaluation](#evaluation)
- [Discussion](#discussion)
- [Results](#results)

<!-- markdown-toc end -->


# Introduction

The computational neuroscience literature abounds with models of individual brain structures, such as the hippocampus, basal ganglia, thalamus, and various cortical areas — from visual to prefrontal. These models typically aim to explain specific functions attributed to each structure. For instance, the basal ganglia are often modeled in the context of decision-making, while the hippocampus is associated with episodic memory and spatial navigation through place cells. However, such models are usually highly abstract and simplified, often relying on a small number of over-engineered neurons and synapses, dealing mostly with abstract inputs and outputs. Consequently, despite decades of work, we still lack an integrated, functional mini-brain — a synthetic neural system capable of performing even simple, continuous embodied tasks in a simulated environment.

The **BrainCraft Challenge** seeks to address this gap by promoting the development of such systems. Specifically, the challenge invites participants to design a biologically inspired, rate-based neural network capable of solving simple decision tasks. The network must control an agent (or "bot") situated in a continuous, low-complexity environment. The agent’s sole objective is to locate and reach an energy source in order to maintain its viability.

# Tasks

## Timeline

The whole challenge is made of 5 different tasks with increasing complexity. Each new task will be introduced at specific dates and participants will have 2 months to try to complete them.

| #  | Name              |  Start         | End            | Status     |
| -- | ----------------- | -------------- | -------------- | ---------- |
| 1  | [Simple decision] | `JUL 01, 2025` | `AUG 31, 2025` | **Active** |
| 2  | [Cued decision]   | `SEP 01, 2025` | `OCT 31, 2025` | **Active** |
| 3  |                   | `NOV 01, 2025` | `DEC 31, 2025` | Inactive   |
| 4  |                   | `JAN 01, 2026` | `FEB 28, 2026` | Inactive   |
| 5  |                   | `MAR 01, 2026` | `APR 30, 2026` | Inactive   |

[Simple decision]: #task-1-simple-decision
[Cued decision]: #task-2-cued-decision


## Task 1: Simple decision

The environment is a 10x10 square maze with three parallels vertical path as illustrated on Figure 1. The cartesian coordinates inside the maze are normalized such that any position (x,y) ∈ [0,1]×[0,1]. Walls can possess a color c ∈ ℕ⁺ whose semantic is not specified a priori. There exists a single energy source that is located at location 1 or 2 at the start of each run (with equal probability). This location remains constant throughout a run but is unknown to the bot. When the bot goes over a source, its energy level is increased by a specific amount. This lasts until the energy source is depleted, leading eventually to the end of the run by lack of energy. **The energy level of a source is also decreased by a specific amount at each time step**, independently of the presence of the bot.

The optimal strategy involves initially exploring both potential energy source locations, subsequently restricting navigation to the half-loop—traversing the central corridor—that contains the identified energy source.
```
┌─────────────────┐
│                 │
│   ┌──┐   ┌──┐   │  ▲ :   Bot start position & orientation (up)
│   │  │   │  │   │  1/2 : Potential energy source location
│ 1 │  │ ▲ │  │ 2 │         (one or the other)
│   │  │   │  │   │
│   └──┘   └──┘   │
│                 │
└─────────────────┘
```

**Figure 1.** **Schematic of the challenge environment.** The bot begins at the center of the arena, facing upward (indicated by the triangle ▲). At each run, the energy source is located at either position 1 or 2 (but not both). The environment is continuous, and the bot moves at a constant speed. The neural model controls only the agent’s steering — i.e., its change in orientation at each time step.

This first task, that may appear trivial, poses nonetheless a non-trivial challenge for current neuroscience-inspired models because of the hard constraints that have been added (see Methods section below). Success will require combining functional neural dynamics with sensorimotor control in a continuous loop, echoing the principles of embodied cognition.


## Task 2: Cued decision

We reuse the environment from task 1 with some differences. There is now a closed path at either right or left as illustrated on Figure 2. The closed path is indicated by a red color (RR) and the open path is indicated by a blue color (BB). The second difference are the sources 1 and 2 that are simultaneously present. The optimal strategy is thus to follow the initial blue cue, to stick to the initial direction and to ignore the additional blue distractor.

```
┌─────────────────┐
│                 │
│   ┌──┐   ┌──┐   │  ▲ :   Bot start position & orientation (up)
│   │BB│   │RR│   │  1/2 : Potential energy source location
│ 1 │  │ ▲ │  │ 2 │  B: Blue block
│   │  │   │BB└───┤  R: Red block
│   └──┘   └──────┤
│                 │
└─────────────────┘
```

**Figure 2.** **Schematic of the challenge environment.** The bot begins at the center of the arena, facing upward (indicated by the triangle ▲). At each run, an energy source is located on both sides, but only one side allows to move freely while the other is a cul-de-sac.


# Methods

## Bot

The simulated bot is circular with a given radius and evolves at a constant speed. The bot can be only controlled on steering. If it hits a wall, its speed remain constant after the hit. The bot has an initial energy level that is decreased by a given amount after each move or hit. If the energy level of the bot drops to 0, the run is ended.

The bot is equipped with a camera that allows to perceive the environment:

 - **64** distance sensors, spread quasi uniformly between -30° and
          +30° relatively to the heading direction. Each sensor encodes the
          distance to the wall that has been hit.
 - **1** bump sensor indicating if the bot has just hit a wall
 - **1** energy gauge indicating the current level of energy
 - **1** constant value of 1 (might be used for bias)


## Model

The architecture of the model is subject to strict constraints. It consists of an input layer connected to a pool of neurons, which in turn is used to compute the output. The neurons within the pool are leaky rate units characterized by a specific leak rate and activation functions. This structure closely resembles that of an Echo State Network (ESN), and is governed by the following equations:

- **Equation `1`:** X(t+1) = (1-λ)•X(t) + λ•f(W•X(t) + Win•I(t))
- **Equation `2`:** O(t+1) = Wout•g(X(t+1))

where:

   - X(t) ∈ ℝⁿ is the state vector
   - I(t) ∈ ℝᴾ is the input vector
   - O(t) ∈ ℝ  is the output (in degrees)
   - W ∈ ℝⁿˣⁿ  is the recurrent (or inner) weight matrix
   - Win ∈ ℝᴾˣⁿ is the input weight matrix
   - Wout ∈ ℝⁿˣ¹ is the output weight matrix
   - f and g are activation functions (typically hyperbolic tangent and identity, respectively)
   - λ is the leaking rate

The inner weight matrix W can be arbitrarily defined, ranging from purely feedforward structures to random recurrent topologies. The model produces a single scalar output, representing a relative change in heading direction. This output must be constrained to lie within the range [–5°, +5°] relative to the agent’s current orientation.


# Evaluation

The evaluation procedure comprises two distinct phases: a time-constrained [training phase](#training-phase), during which the model parameters are optimized, followed by a [testing phase](#testing-phase), wherein the learned parameters are employed to assess the model’s performance. Upon complete depletion of the agent’s energy reserves, the trial concludes, and a performance score is computed. This score reflects the total distance traversed by the agent during the trial; **higher value indicates higher performance**.

## Training phase

The training phase is limited to a maximum duration of 100 seconds of user time. Within this phase, participants are free to employ any learning paradigm deemed appropriate, including reinforcement learning, supervised or unsupervised learning, evolutionary algorithms, etc. Upon completion of the training phase, the following elements must be returned:

  - `W` ∈ ℝⁿˣⁿ  (inner weight matrix)
  - `Win` ∈ ℝᴾˣⁿ (input weight matrix)
  - `Wout` ∈ ℝⁿˣ¹ (output weight matrix)
  - `λ` ∈ ℝ or ℝⁿ (global or individual leak rates) 
  - `f` and `g` (activation functions)
  - `warmup` duration (bot don't move before warmup period is over)

The training code must be fully self-contained. In particular, it must not rely on any external resources or data (e.g., files or models) that may have been generated during a prior or extended training phase. To ensure reproducibility and compatibility, it is strongly recommended that submissions restrict their dependencies to [NumPy](https://numpy.org), [SciPy](https://scipy.org), and [Matplotlib](https://matplotlib.org) only. Also, if you use an external program to generate a bunch of constant values (e.g. weights), this must be included. In other words, if your programm contains too much constant (let's say above 64), you'll have to explain where do they come from.

**Note**: Any code utilized during the training phase is not available during the testing phase. Consequently, if, for example, a participant employs an external reward signal during training, this signal must be internally generated by the model itself — e.g., derived from the agent’s own state variables such as the rate of change of the energy level.

## Testing phase

During the testing phase, the position and direction of the agent is initialized according to the task. Equations (1) and (2) are iteratively applied until the agent’s energy is fully depleted. The testing phase comprises ten such trials, and the final performance score is computed as the mean distance traveled across all ten runs.

```Python
from challenge import train, evaluate

def model():
    # do something and returns your model
    # you can send intermediate models (yield) as well.
    # See the player_random.py example

    ...
    
    yield default_model
    while True:
        ...
        evaluate(better_model, runs=3)
        yield better_model
    return best_model

evaluate(train(model, timeout=100))
```

## Code helpers

To help with the design and debug of a model, you can use the Bot class that offers
a number of state variables:

 - `radius` (constant, 0.05)
 - `speed` (constant, 0.01)
 - `camera` (constant, fov = 60, resolution = 64)
    - `camera.depths` (wall distances, n=resolution=64)
    - `camera.values` (walls color, n=resolution=64)
 - `position` (initially 0.5, 0.5, task dependent)
 - `direction` (inititally 90° ± 5°, task dependent)
 - `energy` (initially 1)
 - `move_penalty` (constant, 1/1000)
 - `hit_penalty` (constant, 5/1000)

as well as the Environment (task dependent) class that also gives access to energy sources:

 - `energy` (initially 2)
 - `probability` (constant, 1.0, task & source dependent)
 - `quality` (constant, 1.0, task & source dependent)
 - `leak` (constant, 2/1000)
 - `refill` (constant, 5/1000)

These variables can be read (and possibly modified) during training but they won't be accessible during testing (no reading, no writing). To actually move the bot, you need to call the `forward` method. This method first changes the direction of the bot and then move it forward and update the internal state (sensors, hit detection, energy consumption). The evaluation method has also a debug flag that may be helpful to visualize the behavior of your model (see Figure 2).

![](./data/debug.png)

**Figure 2.** **Debug view during evaluation.** The left part is a bird-eye view of the environment where the yellow part is the unique source of energy. The right part is a first-person view build from the set of 64 sensors that is not needed during evaluation (but it might help debug).


# Discussion

- **How to submit?** Make a pull request with your player, assumed performance and (short) description.  I'll then re-run training and evaluation (with a random seed) and add the result to the leader board.

- **Why 1000 neurons?** Since [It Takes Two Neurons To Ride a Bicycle](https://paradise.caltech.edu/~cook/papers/TwoNeurons.pdf), I decided that 1000 neurons should be more than enough to solve such a *simple* task.

- **Why 100 seconds?** Because I want any student to be able to enter the challenge without having access to a supercomputer or a cluster.  A basic laptop should be just fine.

- **Why 10 runs?** Having only 2 possible environments, 10 runs should be enough to give a fair account of your model performance.

- **Why 2 choices?** If you consider the abstraction of the task, there is really two choices at branching points.
 
- **Why no reward?** Because it is easy to generate your own reward signal from the derivation of energy. Is it?

- **Can I run my code on a supercomputer?** Not really because the official evaluation will be ran on my machine (which is a M1 Macbook pro).

- **Can I change the rules?** Of course no.

- **May I propose some rule changes then?** Sure. Just open an issue with the reason you want to change this or that rule.

- **I've found a bug in the simulator** Open an issue and report it, with a fix if possible.

- **Is there a prize?** No, only fame and glory.

- **Do my code needs to be open source?** Absolutely. BSD or GPL

- **Can I use an external program for writing my training function?**  It depends. If the resulting training function is made mostly of a bunch of generated weigths, then no. If you ask help from a generative AI, then ok.
 

# Results

Here is the current leader board. If you think you can do better, make
A pull request with your player. I'll evaluate it and add a line
below.

## Task 1

Author        | Date       | File           | Score                  | Seed   | Description
------------- | ---------- | -------------- | -----------------------|------- | -------------------------
[@rougier]    | 07/06/2025 | [manual.py]    | **15.00** (single run) | None   | Manual player (reference)
[@rougier]    | 24/07/2025 | [random.py]    | **1.46** ±  0.54       | 12345  | Stupid random bot
[@tjayada]    | 31/07/2025 | [evolution.py] | **12.70** ± 0.43       | 78     | [Genetic algorithm]
[@vforch]     | 21/08/2025 | [simple.py]    | **13.71** ± 0.46       | 78     | [Handcrafted weights 1]
[@snowgoon88] | 28/08/2025 | [switcher.py]  | **11.39** ± 3.56       | 78     | [Handcrafted weights 2]
[@vforch],[@snowgoon88] | 31/08/2025 | [switcher_alt.py]  | **14.71** ± 0.46   | 78     | [Handcrafted weights 3]

[@rougier]: https://github.com/rougier
[@tjayada]: https://github.com/tjayada
[@snowgoon88]: https://github.com/@snowgoon88
[@vforch]: https://github.com/vforch
[simple.py]:  ./braincraft/player_simple.py
[random.py]: ./braincraft/player_random.py
[manual.py]: ./braincraft/player_manual.py
[switcher.py]:  ./braincraft/player_switcher.py
[switcher_alt.py]:  ./braincraft/player_switcher_alt.py
[evolution.py]: ./braincraft/player_evolution.py
[Genetic algorithm]: https://github.com/rougier/braincraft/pull/5
[Handcrafted weights 1]: https://github.com/rougier/braincraft/pull/7
[Handcrafted weights 2]: https://github.com/rougier/braincraft/pull/8
[Handcrafted weights 3]: https://github.com/rougier/braincraft/pull/8#issuecomment-3238121164
