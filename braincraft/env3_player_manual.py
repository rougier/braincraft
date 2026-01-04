# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
"""
Evaluation of the performance of a manual (human) player.
"""
import sys
import numpy as np
   
def on_press(event):
    """ Change bot direction with lefT/right arrows """
    global move
    sys.stdout.flush()
    if move:
        if event.key == 'left':
            bot.direction += np.radians(5)
        elif event.key == 'right':
            bot.direction -= np.radians(5)
        move = False        

def update(frame=0):
    """ Update display, bot and stats. """
        
    global anim, bot, graphics, move, distance, environment
    
    position = bot.position
    energy = bot.energy
    bot.forward(0, environment)
    move = True
    distance += np.linalg.norm(position - bot.position)
    bot.camera.render(bot.position, bot.direction,
                      environment.world, environment.colormap)
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
        anim.event_source.stop()

    graphics["camera"].set_data(bot.camera.framebuffer)

    
if __name__ == "__main__":
    from bot import Bot
    from environment_3 import Environment

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection

    environment = Environment()
    bot         = Bot()
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

    print("The manual player is controlled by left and right arrows")
    print("Since the human player (e.g. you) has a bird eye view, this serves as an ideal reference")

    move = True
    distance = 0.0

    fig.canvas.mpl_connect('key_press_event', on_press)
    anim = FuncAnimation(fig, update, frames=360, interval=60, repeat=True)
    plt.show()
    print(f"Final score: {distance:.2f} (single run)")
    sys.stdout.flush()
