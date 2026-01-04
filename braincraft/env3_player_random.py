# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
"""
Example and evaluation of the performances of a random player.
"""
from bot import Bot
from environment_3 import Environment
    
def identity(x):
    return x

def random_player():
    """Random players building"""

    np.random.seed(1)
    bot = Bot()

    # Fixed parameters
    n = 1000
    p = bot.camera.resolution
    warmup = 0
    f = np.tanh
    g = np.tanh
    leak = 0.85

    # Random search for best model for 5 tries  (not efficient at all)
    model_best = None
    score_best = -1    

    for i in range(5):
        Win  = np.random.uniform(-1,1, (n,2*p+3))
        W = np.random.uniform(-1,1, (n,n))*(np.random.uniform(0,1, (n,n)) < 0.1)
        Wout = 0.1*np.random.uniform(-1, 1, (1,n))
        model = Win, W, Wout, warmup, leak, f, g

        # Evaluate model on 3 runs
        score_mean, score_std = evaluate(model, Bot, Environment, runs = 3, debug=False)
        
        if score_mean > score_best:
            score_best = score_mean
            model_best = Win, W, Wout

        # Return temporary best result
        yield *model_best, warmup, leak, f, g


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import numpy as np    
    from challenge_2 import train, evaluate

    seed = 12345
    
    # Training (100 seconds)
    np.random.seed(seed)
    print(f"Starting training for 100 seconds (user time)")
    model = train(random_player, timeout=100)

    # Evaluation
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")

