# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
import time
import numpy as np
from tqdm import tqdm
from camera import Camera

# from bot import Bot
# from environment import Environment

def train(func, timeout=100.0):
    """
    Runs the given training function with a user-time timeout and
    returns the last result. The training function is supposed to yield
    temporary results or else, training can be terminated without any results.
    """

    start = time.process_time()
    elapsed = time.process_time() - start
    last_result = None

    with tqdm(total=timeout, unit="s",
              bar_format="Elapsed: {bar}| {elapsed}s (real-time)") as progress:

        # Run training until timeout or when no more training needed.
        for i, result in enumerate(func()):
            elapsed = time.process_time() - start
            if elapsed > timeout:
                progress.update(timeout-progress.n)
                break
            else:
                progress.update(elapsed - progress.n)
                last_result = result

        # Timeout has been reached or training stops early
        progress.close()
        if elapsed > timeout:
            overshoot = 100*(elapsed/timeout - 1)
            print(f"Training stopped at {elapsed:.2f}s "
                  f"({i} iterations, overshoot by {overshoot:.1f}%)")
        else:
            undershoot = 100*(1-elapsed/timeout)
            print(f"Training stopped early at {elapsed:.2f}s "
                  f"({i} iterations, undershoot by {undershoot:.1f}%)")

    return last_result



def evaluate(model, Bot, Environment, runs=10, seed=None, debug=False):
    """Evaluate a model with the given number of runs.

    Parameters
    ----------
    model : list    
      model should be a W_in, W, W_out, warmup, leak, f, g list

    Environment: class
      Environment class to use for evaluation
    
    Bot: class
      Bot class to use for evaluation
    
    runs : int
      Number of runs to evaluate
     
    debug : boolean
      Whether to display animation (slow)
    
    Returns
    =======
    Mean score (float) and stndard deviation over the given number of runs.
    """

    if seed is None:
        seed = np.random.randint(10_000_000)
    
    if debug:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.animation import FuncAnimation
        from matplotlib.collections import LineCollection

        environment = Environment()
        world = environment.world
        world_rgb = environment.world_rgb
        
        fig = plt.figure(figsize=(10,5))
        ax1 = plt.axes([0.0,0.0,1/2,1.0], aspect=1, frameon=False)
        ax1.set_xlim(0,1), ax1.set_ylim(0,1), ax1.set_axis_off()
        ax2 = plt.axes([1/2,0.0,1/2,1.0], aspect=1, frameon=False)
        ax2.set_xlim(0,1), ax2.set_ylim(0,1), ax2.set_axis_off()

        graphics = {
            "topview" : ax1.imshow(environment.world_rgb, interpolation="nearest", origin="lower",
                                   extent = [0.0, world.shape[1]/max(world.shape),
                                             0.0, world.shape[0]/max(world.shape)]),
            "bot" : ax1.add_artist(Circle((0,0), 0.05,
                                          zorder=50, facecolor="white", edgecolor="black")),
            "rays" : ax1.add_collection(LineCollection([], color="C1", linewidth=0.5, zorder=30)),
            "hits" :  ax1.scatter([], [], s=1, linewidth=0, color="black", zorder=40),
            "camera" : ax2.imshow(np.zeros((1,1,3)), interpolation="nearest",
                                  origin="lower", extent = [0.0, 1.0, 0.0, 1.0]),
            "energy" : ax2.add_collection(
                LineCollection([[(0.1, 0.1),(0.9, 0.1)],
                                [(0.1, 0.1),(0.9, 0.1)],
                                [(0.1, 0.1),(0.9, 0.1)]],
                               color=("black", "white", "C1"), linewidth=(20,18,12),
                               capstyle="round", zorder=150)) }

    
    # Unfold model
    W_in, W, W_out, warmup, leak, f, g = model

    scores = []
    seeds = np.random.randint(0, 1_000_000, runs)
    # print(f"Seeds : {seeds}")

    for i in range(runs):
        np.random.seed(seeds[i])
        environment = Environment()

        if debug:
            graphics["topview"].set_data(environment.world_rgb)
        
        bot = Bot()

        # We now take color into account
        n = bot.camera.resolution
        I, X = np.zeros((2*n+3,1)), np.zeros((1000,1))

        distance = 0
        hits = 0
        iteration = 0

        # Initial update
        if debug:
            bot.camera.render(bot.position, bot.direction,
                              environment.world, environment.colormap)
        else:
            bot.camera.update(bot.position, bot.direction,
                              environment.world, environment.colormap)

        # Run until no energy
        while bot.energy > 0:

            energy = bot.energy

            # Distances (the higher, the closer)
            I[:n,0] = 1 - bot.camera.depths

            # Colors
            I[n:2*n,0] = bot.camera.values

            # Hit, energy, bias
            I[2*n:,0] = bot.hit, bot.energy, 1.0
            
            X = (1-leak)*X + leak*f(np.dot(W_in, I) + np.dot(W, X))
            O = np.dot(W_out, g(X))
            
            # During warmup, bot does not move
            if iteration > warmup:
                p = bot.position
                bot.forward(O, environment, debug)
                distance += np.linalg.norm(p - bot.position)
                hits += bot.hit
            iteration += 1

            if debug:
                graphics["rays"].set_segments(bot.camera.rays)
                graphics["hits"].set_offsets(bot.camera.rays[:,1,:])
                graphics["bot"].set_center(bot.position)
                if energy < bot.energy:
                    graphics["energy"].set_color( ("black", "white", "C2") )
                else:
                    graphics["energy"].set_color( ("black", "white", "C1") )        

                if bot.energy > 0:
                    ratio = bot.energy/bot.energy_max
                    graphics["energy"].set_segments([[(0.1, 0.1),(0.9, 0.1)],
                                                     [(0.1, 0.1),(0.9, 0.1)],
                                                     [(0.1, 0.1),(0.1 + ratio*0.8, 0.1)]])
                else:
                    graphics["energy"].set_segments([[(0.1, 0.1),(0.9, 0.1)],
                                                     [(0.1, 0.1),(0.9, 0.1)]])            
                graphics["camera"].set_data(bot.camera.framebuffer)
                bot.camera.render(bot.position, bot.direction,
                              environment.world, environment.colormap)
                plt.pause(1/60)

        scores.append(distance)

    return np.mean(scores), np.std(scores)


