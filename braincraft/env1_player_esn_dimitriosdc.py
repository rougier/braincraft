import numpy as np
import random
from .camera import Camera
from .environment_1 import Environment

# === Nonlinearities (copied from your evolution code) ===
def f(x):
    return np.tanh(x)

def g(x):
    return 1.0 / (1.0 + np.exp(-x))


# ===== Model sizes =====
N_CAMERA = 64           # from Camera.depths
N_META = 3              # [hit, energy, bias]
N_VIS_MEM = 100         # 10 x 10 flattened
N_INPUT = N_CAMERA + N_META + N_VIS_MEM  # 64 + 3 + 100 = 167
N_RESERVOIR = 1000

# ===== ESN utils =====

def make_scaled_reservoir(Wres, spectral_radius=0.9):
    """Scale Wres to the given spectral radius (safe for 1000x1000)."""
    # Power iteration for largest |eig|
    v = np.random.randn(Wres.shape[0], 1)
    for _ in range(50):
        v = Wres @ v
        v_norm = np.linalg.norm(v) + 1e-12
        v /= v_norm
    lam = float(np.linalg.norm(Wres @ v, 2))
    if lam > 0:
        Wres = Wres * (spectral_radius / lam)
    return Wres

def hard_clip(x, lo=-10.0, hi=10.0):
    return np.minimum(np.maximum(x, lo), hi)

# ===== Bot =====

class ESNBot:
    """
    Action head: logistic on mean(state) with energy-gated explore/exploit bias.
    Input vector: [1 - depths(64), hit, energy, 1.0, visual_memory(100)]
    Visual memory is a 10x10 binary map of visited cells (flattened).
    """
    def __init__(self, Win, Wres, grid_wh=(10, 10), explore_energy_thresh=0.6):
        self.camera = Camera()

        # Matrices
        assert Win.shape[1] == N_INPUT, f"Win second dim must be {N_INPUT}, got {Win.shape[1]}"
        assert Wres.shape[0] == Wres.shape[1] == N_RESERVOIR, "Wres must be NxN with N=1000"
        self.Win = Win.astype(np.float64, copy=True)
        self.Wres = make_scaled_reservoir(Wres.astype(np.float64, copy=True), spectral_radius=0.9)

        # Dimensions
        self.n_input = self.Win.shape[1]
        self.n_reservoir = self.Wres.shape[0]

        # State
        self.reservoir_state = np.zeros((self.n_reservoir, 1), dtype=np.float64)
        self.leak = 0.8

        # Kinematics & meta
        self.position = np.array([0.5, 0.5], dtype=np.float64)
        self.direction = np.radians(np.random.uniform(-5, +5))
        self.energy = 1.0
        self.hit = 0

        # Env-compat energy costs (read by env.update(bot))
        self.energy_move = 1 / 1000
        self.energy_hit = 5 / 1000

        # Visual memory
        self.grid_w, self.grid_h = grid_wh
        self.visited = np.zeros((self.grid_h, self.grid_w), dtype=bool)

        # Explore/exploit gating
        self.explore_energy_thresh = explore_energy_thresh
        self._last_action = 1  # start moving

    # ----- Inputs -----

    def _encode_visual_memory(self):
        # Flatten 10x10 -> 100 in row-major
        return self.visited.astype(np.float32).reshape(-1)

    def _update_memory(self, world_shape):
        # Mark current cell as visited
        h, w = world_shape
        x = min(int(self.position[0] * w), w - 1)
        y = min(int(self.position[1] * h), h - 1)
        self.visited[y, x] = True

    def get_input_vector(self):
        # Camera depths in [0..1], convert to “closeness” = 1 - depth
        n = self.camera.resolution  # expect 64
        x = np.zeros((N_INPUT, 1), dtype=np.float64)
        x[:n, 0] = 1.0 - np.nan_to_num(self.camera.depths, nan=1.0, posinf=1.0, neginf=1.0)

        # Meta
        x[n:n+3, 0] = [float(self.hit), float(self.energy), 1.0]

        # Visual memory
        mem = self._encode_visual_memory()
        x[n+3:, 0] = mem  # length 100
        return x

    # ----- Policy -----

    def _policy(self, state_mean):
        """
        Logistic over mean reservoir state, energy-gated:
        high energy -> small bias towards exploring turns;
        low energy -> bias towards forward motion (exploit nearby).
        """
        # Mode gate
        explore_mode = (self.energy >= self.explore_energy_thresh)

        # Bias term nudges the logistic to favor turn while exploring, move while exploiting
        bias = 0.15 if explore_mode else -0.15
        z = state_mean + bias
        p_move = 1.0 / (1.0 + np.exp(-z))
        return 1 if p_move >= 0.5 else 0  # 1: move forward, 0: turn

    # ----- Step -----

    def step(self, env):
        # Clamp
        self.position = np.clip(self.position, 0.0, 1.0 - 1e-8)

        # Sense
        self.camera.update(self.position, self.direction, env.world, env.colormap)
        if (not np.all(np.isfinite(self.camera.depths))) or np.any(self.camera.depths == 0):
            self.camera.depths = np.ones_like(self.camera.depths)

        # Update visual memory for current cell
        self._update_memory(env.world.shape)

        # ESN update
        u = self.get_input_vector()  # (167,1)
        pre = self.Win @ u + self.Wres @ self.reservoir_state
        pre = hard_clip(pre, -25.0, 25.0)
        r = f(pre)  # e.g., tanh
        self.reservoir_state = (1.0 - self.leak) * self.reservoir_state + self.leak * r
        self.reservoir_state = hard_clip(self.reservoir_state, -10.0, 10.0)

        # Action
        state_mean = float(self.reservoir_state.mean())
        action = self._policy(state_mean)

        # Apply action
        if action == 0:
            # explore turn: small randomized turning to avoid lock-in
            self.direction += np.pi / 4.0 + np.radians(np.random.uniform(-5, 5))
        else:
            step = 0.02
            dx = step * np.cos(self.direction)
            dy = step * np.sin(self.direction)
            self.position += np.array([dx, dy])
            self.position = np.clip(self.position, 0.0, 1.0 - 1e-8)

        self._last_action = action
        return action
