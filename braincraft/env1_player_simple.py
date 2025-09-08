# Braincraft challenge entry — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# by Valentin Forch
"""
Example and evaluation of the performances of a random player.
"""

import numpy as np
from bot import Bot
from environment_1 import Environment
from nonlinearities import NonlinearityType, get_nonlinearity_function
    
def custom_eval(Environment, Bot, model, n_min, n_evals=1, t_max=300):
    """Agent evaluation with early stopping if to many wall hits happen"""
    
    score_list, hit_list, energy_list = [], [], []
    for n in range(n_evals):
        
        environment = Environment()
        bot = Bot()
        
        W_in, W, W_out, warmup, leak, f, g = model
        
        # Convert enum types to functions if needed
        if isinstance(f, NonlinearityType):
            f = get_nonlinearity_function(f)
        if isinstance(g, NonlinearityType):
            g = get_nonlinearity_function(g)
        
        n_cam = bot.camera.resolution
        n_inp = W_in.shape[1]
        n_rec = W.shape[0]
        
        # prepare simulation
        I = np.zeros(n_inp)
        X = np.zeros(n_rec)
        O = 0
    
        t = 0
        score = 0
        hits = 0
        for t in range(t_max):            
            if bot.hit:
                score -= 1
                hits += 1
            else:
                score += 1
            
            I[ : n_cam] = 1 - bot.camera.depths
            I[ n_cam :] = bot.hit, bot.energy, 1.0
    
            X = (1 - leak) * X + leak * f(np.dot(W_in, I) + np.dot(W, X))
            O = g(np.dot(W_out, X))
            O = np.clip(O, -5, 5)
            bot.forward(O, environment)
            
            if bot.energy < 0.001 or hits >= 30:
                break
            
        score_list.append(score)
        hit_list.append(hits)
        energy_list.append(bot.energy)
    
        if hits >= 30:
            break
    
    return np.mean(score_list)
    
def simple_player():
    """Aims to build the most simplistic controller which avoids hitting walls. Drives only along the outer lane."""
    bot = Bot()
     
    # network hyperparameters
    leak = 0.95
    f = NonlinearityType.TANH  # Use TANH for the hidden layer activation
    g = NonlinearityType.LINEAR  # Use LINEAR for the output layer
    
    # model size
    n_cam = bot.camera.resolution
    n_inp = n_cam + 3
    n_rec = 1000 
    
    n_min = n_cam - 1  # only using every n_min'th sensor node
        
    score_best = -np.inf
    model_best = None
    while True:
        
        model_values = np.random.uniform(0, 1, 2)
            
        # input parameters
        exc_input = model_values[0]
        thresh_inp = model_values[1]      
        
        # construct the model
        W_in = np.zeros((n_rec, n_inp))
        W = np.zeros((n_rec, n_rec))
        W_out = np.zeros(n_rec)
        
        model = W_in, W, W_out, 0, leak, f, g
    
        # input weights for connecting sensor to steering pop
        for i in range(0, n_cam, n_min):  W_in[i, i] = exc_input
        # constant threshold for "ignoring" walls beyond a certain depth
        W_in[:n_cam:n_min, -1] = -thresh_inp
    
        # output weights
        W_out[:n_cam:n_min] = n_min * np.linspace(-1, 1, len(W_out[:n_cam:n_min]))
        
        # Evaluate model
        score_mean = custom_eval(Environment, Bot, model, n_min)    
        
        if score_mean >= 300: break    
        
        if score_mean > score_best:
            score_best = score_mean
            model_best = model
        yield model_best
        
    yield model
    
if __name__ == "__main__":
    import time
    import numpy as np    
    from challenge import train, evaluate
    
    seed = 78
    np.random.seed(seed)
    
    # Training (100 seconds)
    print(f"Starting training for 100 seconds (user time)")
    model = train(simple_player, timeout=100)
    
    W_all = np.concatenate([np.concatenate([np.array([0]), model[2], np.zeros(67)])[None, :],
                            np.concatenate([np.zeros((1000, 1)), model[1], model[0]], axis=1), 
                            np.zeros((67, 1068))])
    print("Number of 'neurons' used in network (including inputs and output):", 
          sum(((np.sum(np.abs(W_all), 0) > 0) + (np.sum(np.abs(W_all), 1) > 0)) > 0))
    print("Non-zero weights in network:", np.sum(W_all!=0))
    
    # Evaluation
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, runs=10, debug=False, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")

