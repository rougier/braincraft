# Braincraft challenge entry — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# by Alain Dutech
"""
Example and evaluation of the performances of a handcrafter switcher player.
"""

from bot import Bot
from environment_1 import Environment

# ** activation functions ******************************************************
def ReLU(x):
    return np.clip(x, a_min=0, a_max=None)
    
def identity(x):
    return x

# ** define a model ***********************************************************
def switcher_player():
    """An agent with 3 "behaviors", in a kind of subsumption way.
    - go away from things on the right => leads to circling outer perimeter anti-clockwise
    - goaway from close things on the left => avoid abstacle too close
    - if rewarded, take a turn left in the "central corridor"

    NO LEARNING.
    """
    leak = 1.0
    warmup = 1

    bot = Bot()
    dim_cam = bot.camera.resolution
    dim_input = dim_cam + 3
    dim_res = 1000
    # indexes for some input and X neurones
    i_left = 0
    i_right = dim_cam - 1
    i_center_right = dim_cam - 31
    i_nrj = dim_input - 2
    i_bias = dim_input - 1
    i_filter = i_nrj + 10
    i_turn = i_filter + 10
    i_turn_mid = i_turn+10

    # weights
    W_in = np.zeros((dim_res, dim_input)  )
    W = np.zeros((dim_res, dim_res))
    W_out = np.zeros((1,dim_res))

    # TurnLeft (away from SensorRight)
    W_in[1, i_right] = 1.0
    W_in[1, i_bias] = -0.79
    W_out[0, 1] = 100

    # TurnRight (avoid close SensorLeft)
    W_in[i_turn+1, i_left] = 1.0
    W_in[i_turn+1, i_bias] = -0.88
    W_out[0, i_turn+1] = -50

    # detect NRJ increase.
    W_in[i_nrj, i_nrj] = 1.0
    W[i_nrj+1, i_nrj] = 1.0
    W[i_nrj+2, i_nrj] = 1.0
    W[i_nrj+2, i_nrj+1] = - 1.0
    # need a band-filter so as to only detect "small" NRJ bumps
    # (and not the starting bump with value '1')
    W[i_filter, i_nrj+2] = 1.0
    W[i_filter, i_bias]  = -0.004
    W[i_filter+1, i_nrj+2] = 1.0
    # i_filter+2 accumulate and "memorise" reward increase
    W[i_filter+2, i_filter]   = -2.0
    W[i_filter+2, i_filter+1] = 1.0
    W[i_filter+2, i_filter+2] = 1.05
    W[i_filter+2, i_turn_mid+4] = -1000.0  # shut off when end_midturn
    # i_filter+3 is a NOT(i_filter+2)
    W[i_filter+3, i_filter+2] = -1.0
    W[i_filter+3, i_bias] = +0.07

    # "mid_turn" are started when a far away obstacle on i_left has been seen
    # enough time since some time (hence the * 1.05 feedback on i_turn_mid+1)
    # cumulate i_left < 0.40 when rewarded
    W_in[i_turn_mid, i_left] = -1.0
    W_in[i_turn_mid, i_bias] = 0.45
    W[i_turn_mid, i_filter+3] = -10 # do not detect if no reward received yet
    # accumulate diff
    W[i_turn_mid+1, i_turn_mid] = 1.6
    W[i_turn_mid+1, i_turn_mid+1] = 1.07
    W[i_turn_mid+1, i_turn_mid+4] = -1000  # shut off at end of mid_turn
    # check when to start (threshold i_turn_mid+2)
    W[i_turn_mid+2, i_turn_mid+1] = 1.0
    W[i_turn_mid+2, i_bias] = -1.3

    # when to stop "mid_turn"
    # need to know when NOT(mid_turn)
    W[i_turn_mid+3, i_turn_mid+2] = -1.0
    W_in[i_turn_mid+3, i_bias] = 1.3
    # detect if center_right < 43 AND turning with i_turn_mid+2 (i.e i_turn_mid+2 > 0)
    W_in[i_turn_mid+4, i_center_right] = -1
    W_in[i_turn_mid+4, i_bias] = 0.43
    W[i_turn_mid+4, i_turn_mid+3] = -1   # no while NOT mid_turn

    # finaly, apply to output
    W_out[0, i_turn_mid+2] = 100

    f = lambda x: ReLU( x )
    g = identity
    model = W_in, W, W_out, warmup, leak, f, g

    yield model
    return model


if __name__ == "__main__":
    import time
    import numpy as np    
    from challenge_1 import train, evaluate
    
    seed = 78
    np.random.seed(seed)
    
    # Training (100 seconds)
    timeout_train = 100
    print(f"Starting training for {timeout_train} seconds (user time)")
    model = train(switcher_player, timeout=timeout_train)

    # Evaluation
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, runs=10, debug=False, seed=seed )
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")

