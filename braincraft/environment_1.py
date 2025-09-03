# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
import numpy as np
from typing import Tuple, List, Dict, ClassVar
from dataclasses import dataclass, field, fields


@dataclass
class Source:
    """
    An energy source is a place inside an arena where the bot can
    get energy refill whose quality, quantity and probabability
    depends on the source attributes. Each source is identified
    by a negative number (identity).
    """

    identity: int                       # Identity of the source (mandatory & negative)
    energy: int         = 2             # Initial energy level
    probability: float  = 1.0           # Probability of refill
    quality: int        = 1             # Quality of the source (1, 2, or 3)
    leak: float         = 2/1000        # Source leak per iteration
    refill: float       = 5/1000        # Refill amount for the bot

    def update(self):
        """Update source energy level"""
        
        self.energy = max(0, self.energy - self.leak)
    
    def get_refill(self):
        """Return the amount of refill and decrease energy level
        accordingly"""

        if np.random.uniform(0,1) < self.probability:
            refill = min(self.refill, self.energy)
            self.energy -= refill
            return refill
        return 0

    
@dataclass
class Environment:
    """
    An environment is made of walls (> 0) and energy sources (< 0)
    """

    # 2D world block description
    world: np.ndarray = field(default_factory=lambda: np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1,-1,-1, 1, 0, 0, 1,-2,-2, 1],
        [1,-1,-1, 1, 0, 0, 1,-2,-2, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=int))

    # Class variable for the color map (shared by all instances)
    colormap: ClassVar[Dict[int, Tuple[int, int, int]]] = {
        -4 : [255, 255, 150], # source color
        -3 : [200, 200, 255], # ground (light blue)
        -2 : [255, 255, 255], # ground (normal)
        -1 : [100, 100, 255], # sky
         0 : [255, 255, 255], # empty
         1 : [200, 200, 200], # wall (light)
         2 : [100, 100, 100], # wall (dark)
         3 : [255, 255,   0], # wall (yellow)
         4 : [  0,   0, 255], # wall (blue)
         5 : [255,   0,   0], # wall (red)
         6 : [  0, 255,   0], # wall (green)
    }


    def update(self, bot):
        """Update the environment, refill bot if on a source and
        decrease bot energy by move and hit penalty"""

        # Source leak
        self.source.update()

        # Bot refill bonus
        x, y = bot.position
        cell_size = 1/max(self.world.shape)
        cx,cy = int(x/cell_size), int(y/cell_size)

        if self.world[cy,cx] == self.source.identity:
            bot.energy += self.source.get_refill()

        # Bot move and hit penalties
        bot.energy -= bot.energy_move
        if bot.hit:
            bot.energy -= bot.energy_hit
        
    
    def __post_init__(self):

        # Initialize RGB world
        self.world_rgb = np.zeros((*self.world.shape, 3), dtype=np.uint8)

        # Apply colormap to all blocks (default)
        for value, color in self.colormap.items():
            self.world_rgb[self.world == value] = color

        # Create a random source 
        sources = np.unique(self.world[self.world < 0])
        if len(sources) > 0:
            index = np.random.randint(0, len(sources))
            source_id = sources[index]
            self.source = Source(identity=source_id)

            # Hide all sources except the selected one
            for sid in sources:
                self.world_rgb[self.world == sid] = self.colormap[0]

            # Highlight the selected source with cmap[-4] (blue)
            self.world_rgb[self.world == source_id] = self.colormap[-4]

            
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    env = Environment()
    for row in env.world_rgb:
        for r, g, b in row:
            print(f"\033[38;2;{r};{g};{b}m██", end="")
        print("\033[0m")  # Reset color at end of line
