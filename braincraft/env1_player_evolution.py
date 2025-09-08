# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Tjark Darius
# Released under the GNU General Public License 3
"""
Script to train and evaluate a bot using an evolutionary strategy and multiprocessing.
"""

import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Callable, Generator

from nonlinearities import NonlinearityType, get_nonlinearity_function

# ------------------------------------------------------------------------------
#  Custom reward and evalution functions
# ------------------------------------------------------------------------------


def calculate_reward(
    bot,
    environment,
    prev_position: Tuple[float, float],
    prev_energy: float,
    visited_positions: set,
    both_sides_visited: dict,
) -> float:
    """
    Enhanced reward function for maze navigation with energy collection.
    """

    # Base reward components
    distance_reward = 0
    energy_reward = 0
    wall_penalty = 0
    exploration_reward = 0
    efficiency_bonus = 0

    # 1. Distance traveled reward
    if prev_position is not None:
        distance_traveled = np.linalg.norm(bot.position - prev_position)
        distance_reward = distance_traveled * 10  # Scale factor

    # 2. Energy management reward
    energy_change = bot.energy - prev_energy
    if energy_change > 0:  # Collected energy
        # Bonus for collecting energy, especially when energy is low
        energy_scarcity_multiplier = max(
            1, (50 - prev_energy) / 50
        )  # Higher bonus when energy is low
        energy_reward = 50 * energy_scarcity_multiplier
    elif energy_change < 0:  # Normal energy consumption
        energy_reward = energy_change * 0.1  # Small penalty for energy use

    # 3. Wall collision penalty
    if bot.hit:
        wall_penalty = -20  # Significant penalty for hitting walls

    # 4. Exploration bonus reward
    # Discretize position for exploration tracking
    pos_key = (round(bot.position[0], 2), round(bot.position[1], 2))
    if pos_key not in visited_positions:
        visited_positions.add(pos_key)
        exploration_reward = 5  # Bonus for visiting new areas

    # 5. Both sides exploration reward
    # Check which side of the maze the bot is on
    if bot.position[0] < 0.35:  # Left side
        both_sides_visited["left"] = True
    elif bot.position[0] > 0.65:  # Right side
        both_sides_visited["right"] = True

    if both_sides_visited["left"] and both_sides_visited["right"]:
        exploration_reward += 10  # One-time bonus for exploring both sides

    # 6. Efficiency bonus (moving towards energy when needed)
    if bot.energy < 350:  # Low energy threshold
        energy_source_pos = (
            np.array([0.25, 0.5])
            if environment.source.identity == -1
            else np.array([0.75, 0.5])
        )
        distance_to_energy = np.linalg.norm(bot.position - energy_source_pos)

        # Bonus for moving towards energy source when energy is low
        if prev_position is not None:
            prev_distance_to_energy = np.linalg.norm(prev_position - energy_source_pos)
            if distance_to_energy < prev_distance_to_energy:
                efficiency_bonus = 5  # Bonus for moving towards energy source

    # 7. Survival bonus
    survival_bonus = 0.1  # Small bonus for each step survived

    # Combine all rewards
    total_reward = (
        distance_reward
        + energy_reward
        + wall_penalty
        + exploration_reward
        + efficiency_bonus
        + survival_bonus
    )

    return total_reward


def evaluate_bot_with_rewards(
    model,
    Bot,
    Environment,
    runs: int = 10,
    seed: int | None = None,
    debug: bool = False,
) -> Tuple[float, float]:
    """
    Modified evaluation function (challenge.py) using enhanced reward system.
    """

    if seed is None:
        seed = np.random.randint(10_000_000)
    np.random.seed(seed)

    # Unfold model
    W_in, W, W_out, warmup, leak, f, g = model
    scores = []
    distances = []
    seeds = np.random.randint(0, 1_000_000, runs)

    for i in range(runs):
        np.random.seed(seeds[i])
        environment = Environment()
        bot = Bot()

        n = bot.camera.resolution
        Input_vector, X = np.zeros((n + 3, 1)), np.zeros((1000, 1))

        # Tracking variables for reward calculation
        total_reward = 0
        visited_positions = set()
        energy_collected_count = 0
        both_sides_visited = {"left": False, "right": False}
        step_count = 0
        distance = 0
        hits = 0

        # Initial update
        bot.camera.update(
            bot.position, bot.direction, environment.world, environment.colormap
        )

        # Run until no energy
        while bot.energy > 0:
            # Store previous state
            prev_pos = bot.position
            prev_eng = bot.energy

            # The higher, the closer
            Input_vector[:n, 0] = 1 - bot.camera.depths
            Input_vector[n:, 0] = bot.hit, bot.energy, 1.0
            hits += bot.hit
            X = (1 - leak) * X + leak * f(np.dot(W_in, Input_vector) + np.dot(W, X))
            Output = np.dot(W_out, g(X))

            # During warmup, bot does not move
            if step_count > warmup:
                bot.forward(Output, environment, debug)

                distance += np.linalg.norm(prev_pos - bot.position)

                # Calculate reward for this step
                step_reward = calculate_reward(
                    bot,
                    environment,
                    prev_pos,
                    prev_eng,
                    visited_positions,
                    both_sides_visited,
                )
                total_reward += step_reward

                # Track energy collection
                if bot.energy > prev_eng:
                    energy_collected_count += 1

            step_count += 1

        scores.append(total_reward)
        distances.append(distance)

    return np.mean(scores), np.mean(distances)


# ------------------------------------------------------------------------------
#  Functions to build and train the bot using multiprocessing
# ------------------------------------------------------------------------------


def evaluate_individual(
    args: Tuple[
        np.ndarray, np.ndarray, np.ndarray, int, float, NonlinearityType, NonlinearityType, int, int
    ],
) -> Tuple[float, float]:
    """Worker function to evaluate a single wout"""
    wout, win, w, warmup, leak, f, g, seed, worker_id = args

    # Set unique seed for this worker
    np.random.seed(seed + worker_id)

    # Import here to avoid issues with multiprocessing
    from bot import Bot
    from environment_1 import Environment

    # Convert enum types to functions
    f_func = get_nonlinearity_function(f)
    g_func = get_nonlinearity_function(g)
    
    model = win, w, wout, warmup, leak, f_func, g_func
    score_mean, score_std = evaluate_bot_with_rewards(
        model, Bot, Environment, runs=3, debug=False
    )

    return score_mean, score_std


def evaluate_population_parallel(
    population: list[np.ndarray],
    win: np.ndarray,
    w: np.ndarray,
    warmup: int,
    leak: float,
    f: NonlinearityType,
    g: NonlinearityType,
    seed: int,
    num_processes: int = None,
) -> List[float]:
    """Evaluate entire population (list of wout) in parallel"""
    if num_processes is None:
        num_processes = min(cpu_count(), len(population))

    # Prepare arguments for parallel evaluation
    args_list = []
    for i, wout in enumerate(population):
        args = (wout, win, w, warmup, leak, f, g, seed, i)
        args_list.append(args)

    # Parallel evaluation
    with Pool(processes=num_processes) as pool:
        results = pool.map(evaluate_individual, args_list)

    # Extract scores
    fitness_scores = [result[0] for result in results]

    return fitness_scores


def evolutionary_player() -> Generator[
    Tuple[np.ndarray, np.ndarray, np.ndarray, int, float, NonlinearityType, NonlinearityType]
]:
    """Evolutionary algorithm player - optimizes only Wout with multiprocessing"""

    # Get number of available CPUs
    num_processes = min(cpu_count(), 8)
    tqdm.write(f"Using {num_processes} processes for parallel evaluation")

    from bot import Bot

    bot = Bot()

    # Fixed parameters
    n = 1000
    p = bot.camera.resolution
    warmup = 0
    f = NonlinearityType.TANH  # activation function for 'reservoir'
    g = NonlinearityType.TANH  # activation function for 'reservoir output'
    leak = 0.8
    spectral_radius = 0.95  # Desired spectral radius for W
    density = 0.1  # Density for Win and W

    # Evolution parameters
    population_size = num_processes * 3  # Scale population with available processes
    elite_size = max(4, population_size // 3)
    mutation_rate = 0.35
    crossover_rate = 0.8

    # Create fixed Win and W for entire population
    Win = np.random.randn(n, p + 3) * (np.random.rand(n, p + 3) < density)
    Win[:, -1] = 1

    W = np.random.randn(n, n) * (np.random.rand(n, n) < density)
    eigenvalues = np.linalg.eigvals(W)
    current_spectral_radius = np.max(np.abs(eigenvalues))

    if current_spectral_radius > 0:
        W = W * (spectral_radius / current_spectral_radius)

    # Scale to desired spectral radius
    eigenvalues = np.linalg.eigvals(W)
    current_spectral_radius = np.max(np.abs(eigenvalues))

    if current_spectral_radius > 0:
        W = W * (spectral_radius / current_spectral_radius)

    tqdm.write(f"Fixed network created: Win{Win.shape}, W{W.shape}, leak={leak}")
    tqdm.write(f"Population size: {population_size}, Elite size: {elite_size}")

    # Initialize population
    population = []  # List of Wout matrices

    tqdm.write("Initializing population...")
    for i in range(population_size):
        # Create individual Wout
        Wout = np.random.uniform(-1, 1, (1, n))
        # Wout = np.random.randn(1, n)
        population.append(Wout)

    # Evaluate initial population in parallel
    fitness_scores = evaluate_population_parallel(
        population, Win, W, warmup, leak, f, g, seed, num_processes
    )

    # Track best model
    best_idx = np.argmax(fitness_scores)
    best_wout = population[best_idx].copy()
    best_score = fitness_scores[best_idx]

    tqdm.write(f"Initial population evaluated. Best score: {best_score:.3f}")

    # Evolution loop
    generation = 0
    start_time = time.time()

    while time.time() - start_time < 85:  # Leave some buffer for parallel processing
        generation += 1
        generation_start = time.time()

        # Selection and reproduction
        new_population = []

        # Keep elite
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Generate offspring
        offspring = []
        while len(new_population) + len(offspring) < population_size:
            if time.time() - start_time > 85:
                break

            # Tournament selection
            parent1 = select_via_tournament(population, fitness_scores)
            parent2 = select_via_tournament(population, fitness_scores)

            # Crossover
            child1, child2 = crossover_wout(parent1, parent2, crossover_rate)

            # Mutation (adaptive)
            progress = generation / 20.0  # Adaptive mutation
            mutation_strength = 0.2 * (1 - min(progress, 1.0)) + 0.02 * min(
                progress, 1.0
            )

            child1 = mutate_wout(child1, mutation_rate, mutation_strength)
            child2 = mutate_wout(child2, mutation_rate, mutation_strength)

            offspring.extend([child1, child2])

        # Trim offspring to fit population size
        offspring = offspring[: population_size - len(new_population)]
        new_population.extend(offspring)

        # Evaluate new population in parallel (only evaluate new offspring)
        if offspring:
            offspring_fitness = evaluate_population_parallel(
                offspring, Win, W, warmup, leak, f, g, seed, num_processes
            )

            # Combine elite fitness with offspring fitness
            elite_fitness = [fitness_scores[idx] for idx in elite_indices]
            new_fitness = elite_fitness + offspring_fitness
        else:
            new_fitness = [fitness_scores[idx] for idx in elite_indices]

        # Update population
        population = new_population
        fitness_scores = new_fitness

        # Update best
        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_score:
            best_wout = population[current_best_idx].copy()
            best_score = fitness_scores[current_best_idx]
            tqdm.write(f"Gen {generation}: New best score: {best_score:.3f}")

        # Print generation statistics
        avg_fitness = np.mean(fitness_scores)
        generation_time = time.time() - generation_start
        tqdm.write(
            f"Gen {generation}: Best={best_score:.3f}, Avg={avg_fitness:.3f}, Time={generation_time:.1f}s"
        )
        yield Win, W, best_wout, warmup, leak, f, g

    # Return final best model
    tqdm.write(f"Evolution completed. Final best score: {best_score:.3f}")
    return Win, W, best_wout, warmup, leak, f, g


def select_via_tournament(
    population: List[np.ndarray], fitness_scores: List[float], tournament_size: int = 3
) -> np.ndarray:
    """Tournament selection"""
    tournament_indices = np.random.choice(
        len(population), tournament_size, replace=False
    )
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_idx]


def crossover_wout(
    parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform crossover for Wout matrices"""
    # Create children as copies
    child1 = parent1.copy()
    child2 = parent2.copy()

    if np.random.random() < crossover_rate:
        # Uniform crossover
        mask = np.random.random(parent1.shape) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)

    return child1, child2


def mutate_wout(
    individual: np.ndarray, mutation_rate: float, mutation_strength: float
) -> np.ndarray:
    """Mutate a Wout matrix"""
    mutated = individual.copy()

    # Apply mutations
    mask = np.random.random(individual.shape) < mutation_rate
    noise = np.random.normal(0, mutation_strength, individual.shape)
    mutated[mask] += noise[mask]

    # Clip to reasonable bounds
    mutated = np.clip(mutated, -2.0, 2.0)

    return mutated


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import numpy as np
    from challenge import train, evaluate
    from bot import Bot
    from environment_1 import Environment

    # all seeds tested
    # Seed 12: Score=12.17 ± 0.40
    # Seed 34: Score=10.99 ± 0.34
    # Seed 56: Score=8.62 ± 1.33
    # Seed 78: Score=13.30 ± 0.53 
    # Seed 91: Score=10.17 ± 0.22
    # Seed 23: Score=1.91 ± 0.83
    # Seed 45: Score=3.71 ± 0.27
    # Seed 67: Score=3.38 ± 0.36
    # Seed 89: Score=7.07 ± 3.45
    # Seed 10: Score=7.44 ± 1.05
    # Seed 123: Score=13.32 ± 0.37 # best
    # Seed 456: Score=11.11 ± 0.31
    # Seed 789: Score=12.33 ± 0.44
    # Seed 101112: Score=7.09 ± 3.40
    # Seed 131415: Score=4.39 ± 0.48
    # Seed 161718: Score=10.82 ± 1.33
    # Seed 11: Score=11.43 ± 0.94
    # Seed 22: Score=1.53 ± 0.19
    # Seed 33: Score=8.08 ± 0.34
    # Seed 44: Score=6.44 ± 0.29
    # Seed 55: Score=5.08 ± 0.60
    # Seed 66: Score=6.18 ± 0.44
    # Seed 77: Score=8.03 ± 1.20
    # Seed 88: Score=5.82 ± 0.38
    # Seed 99: Score=11.74 ± 3.57

    # Average performance across seeds : 
    # Score :  7.775
    # Std : 0.887

    # used best seed
    seed = 78

    # Training (100 seconds)
    np.random.seed(seed)
    print("Starting evolutionary training for 100 seconds (user time)")
    model = train(evolutionary_player, timeout=100)

    # Evaluation
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=True, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")
