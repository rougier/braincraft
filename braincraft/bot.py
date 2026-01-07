# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
import numpy as np
from camera import Camera
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any


@dataclass
class Bot:
    """ A circular bot with a camera and an energy gauge"""

    radius: float                  = 0.05
    position: Tuple[float, float]  = (0.5, 0.5)
    direction: float               = np.radians(90)
    speed: float                   = 0.01
    camera: Camera                 = Camera(fov=60, resolution=64)
    hit: int                       = 0
    energy: int                    = 1
    energy_min: int                = 0
    energy_max: int                = 1
    energy_move: float             = 1/1000
    energy_hit: float              = 5/1000


    def __post_init__(self):

        # Adding noise to initial direction +/- 5°
        self.direction += np.radians(np.random.uniform(-5, +5))
        

    
    def is_legal(self, position, environment):
        """
        Check whether a circular bot with given radius and position
        is colliding with any wall in the provided environment.

        Parameters
        ----------

        position : tuple of float
            (x, y) position of the robot center in normalized [0, 1] coordinates.

        environment: Environment
            Environment to consider for the placement

        Returns
        -------
        bool
            True if any part of the robot overlaps a wall, False otherwise.
        """

        world = environment.world
        x, y = position
        h, w = world.shape
        cell_size = 1/max(world.shape)
        radius = 1.05*self.radius
        imin = max(0, int((x-radius)/cell_size))
        imax = min(w-1, int((x+radius)/cell_size)) + 1
        jmin = max(0, int((y-radius)/cell_size))
        jmax = min(h-1, int((y+radius)/cell_size)) + 1    
        for j in range(jmin, jmax):
            for i in range(imin, imax):
                if world[j, i] == 1:
                    cmin = np.array([i,j])*cell_size
                    cmax = cmin + cell_size
                    closest = np.maximum(cmin, np.minimum(position, cmax))
                    if np.linalg.norm(closest - position) <= radius:
                        return False
        return True


    def move_to(self, position, environment):
        """
        Move the bot toward a positon using discrete steps and
        stop at first collision givne the provided environment

        Parameters
        ----------

        positon : tuple of float
            Target position (x, y) in normalized [0, 1] coordinates.

        environment: Environment
            Environment to consider for the move

        Returns
        -------

        position : tuple of float
            Final position (x, y) before collision or at target.

        collision : bool
            True if a collision occurred during movement.
        """

        origin = np.array(self.position)
        target = np.array(position)
        direction = target - origin
        epsilon = 1e-4 # precision, this is necessary to avoid floating point problems.
        if np.allclose(direction, 0):
            return tuple(origin), False
        n = int(np.linalg.norm(direction) / epsilon)
        P = origin + np.arange(n).reshape(-1,1)*epsilon*direction
        P[-1] = position
        for i in range(n):
            if not self.is_legal(P[i], environment):
                return P[max(0,i-1)], True
        return position, False

    def forward(self, dtheta, environment, debug=False):
        """Move the bot by first re-orienting it by dtheta (degrees)
        and then making it to move forward.in the provided environment.

        Parameters
        ----------
        dtheta : float
          Change in direction in degrees, must be between -5° and +5°

        environment: Environment
            Environment to consider for the move

        debug: boolean
          Whether to render 3rd person view (slower)

        Returns
        -------
        
        (energy, hit, distance sensors, color sensors)
        """

        # If there is no energy left, run is ended
        if self.energy <= 0:
            return None
                
        # Try to move in the provided direction
        self.direction += max(np.radians(-5), min(np.radians(+5), float(dtheta.item())))
        T = np.array([np.cos(self.direction), np.sin(self.direction)])
        self.position, self.hit = self.move_to(self.position + T*self.speed, environment)
        
        # Update bot energy
        environment.update(self)

        # Update camera (with debug or not)
        if debug:
            self.camera.update(self.position,
                               self.direction,
                               environment.world,
                               environment.colormap)
        else:
            self.camera.render(self.position,
                               self.direction,
                               environment.world,
                               environment.colormap)

        # Return inputs as (energy, hit, depths, colors)
        return self.energy, self.hit, self.camera.depths, self.camera.values
