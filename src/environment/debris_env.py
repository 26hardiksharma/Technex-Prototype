"""
Orbital Debris Collision Avoidance Gymnasium Environment

This environment simulates a 2D LEO scenario where a satellite must autonomously
avoid a piece of orbital debris using onboard decision-making (edge AI).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from ..dynamics.orbital_mechanics import HillEquationsPropagator, ScenarioGenerator
from .reward_functions import RewardConfig


class OrbitalDebrisEnv(gym.Env):
    """
    Gymnasium environment for autonomous debris collision avoidance.
    
    Observation Space:
        - Relative position (x, y): debris position in LVLH frame
        - Relative velocity (vx, vy): closure rate
        - Time to closest approach: estimated TTCA
        - Remaining fuel: available Δv budget
        - Previous action flag: whether last step had thrust
        
    Action Space:
        Discrete (5 actions):
            0: No thrust (coast)
            1: +Δv radial (away from Earth)
            2: -Δv radial (toward Earth)
            3: +Δv along-track (forward)
            4: -Δv along-track (backward)
        
    Reward:
        - Large negative for collision: -1000
        - Small negative for fuel use: -0.01 * Δv
        - Small positive for safe distance: +0.1 per step
        - Terminal bonus for safe pass: +50 to +100
        
    Episode Termination:
        - Collision: distance < collision_radius
        - Safe pass: debris behind satellite and distance > safe_distance
        - Fuel depletion: remaining_fuel <= 0
        - Max steps: prevent infinite episodes
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(
        self,
        dt: float = 10.0,
        max_steps: int = 300,
        collision_radius: float = 75.0,
        safe_distance: float = 500.0,
        initial_fuel: float = 10.0,
        thrust_magnitude: float = 0.05,
        reward_type: str = 'dense',
        scenario_type: str = 'random',
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the orbital debris avoidance environment.
        
        Args:
            dt: Time step for simulation (seconds)
            max_steps: Maximum episode length
            collision_radius: Collision threshold (meters)
            safe_distance: Safe separation distance (meters)
            initial_fuel: Starting fuel budget (m/s total Δv)
            thrust_magnitude: Thrust per action (m/s^2 acceleration)
            reward_type: Reward function type ('dense', 'sparse', 'fuel_aware', 'potential')
            scenario_type: Initial condition type ('random', 'head_on', 'crossing')
            render_mode: Rendering mode ('human', 'rgb_array', None)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.dt = dt
        self.max_steps = max_steps
        self.collision_radius = collision_radius
        self.safe_distance = safe_distance
        self.initial_fuel = initial_fuel
        self.thrust_magnitude = thrust_magnitude
        self.scenario_type = scenario_type
        self.render_mode = render_mode
        
        # Initialize dynamics propagator (LEO ~400km altitude)
        self.propagator = HillEquationsPropagator(n=0.001027)
        
        # Initialize scenario generator
        self.scenario_gen = ScenarioGenerator(seed=seed)
        
        # Initialize reward function
        self.reward_config = RewardConfig(
            reward_type=reward_type,
            collision_radius=collision_radius,
            safe_distance=safe_distance,
            max_fuel=initial_fuel
        )
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Observation space: [x, y, vx, vy, ttca, fuel, prev_action]
        # Normalize to reasonable ranges for neural network
        self.observation_space = spaces.Box(
            low=np.array([
                -15000.0,  # x position (m)
                -15000.0,  # y position (m)
                -200.0,    # x velocity (m/s)
                -200.0,    # y velocity (m/s)
                0.0,       # time to closest approach (s)
                0.0,       # remaining fuel (m/s)
                0.0        # previous action flag
            ]),
            high=np.array([
                15000.0,   # x position (m)
                15000.0,   # y position (m)
                200.0,     # x velocity (m/s)
                200.0,     # y velocity (m/s)
                1000.0,    # time to closest approach (s)
                initial_fuel,  # remaining fuel (m/s)
                1.0        # previous action flag
            ]),
            dtype=np.float32
        )
        
        # State variables
        self.state = None
        self.remaining_fuel = initial_fuel
        self.step_count = 0
        self.previous_action_taken = False
        self.min_distance_achieved = np.inf
        self.total_fuel_used = 0.0
        
        # Trajectory history for rendering
        self.trajectory = []
        self.debris_trajectory = []
        
    def _get_observation(self) -> np.ndarray:
        """Convert current state to observation vector."""
        x, y, vx, vy = self.state
        
        # Calculate derived features
        ttca = self.propagator.estimate_time_to_closest_approach(self.state)
        ttca = min(ttca, 1000.0)  # Cap at max value
        
        # Construct observation
        obs = np.array([
            x, y, vx, vy,
            ttca,
            self.remaining_fuel,
            float(self.previous_action_taken)
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        distance = self.propagator.calculate_distance(self.state)
        rel_velocity = self.propagator.calculate_relative_velocity(self.state)
        ttca = self.propagator.estimate_time_to_closest_approach(self.state)
        
        return {
            'distance': distance,
            'relative_velocity': rel_velocity,
            'ttca': ttca,
            'remaining_fuel': self.remaining_fuel,
            'fuel_used': 0.0,  # Updated in step()
            'total_fuel_used': self.total_fuel_used,
            'min_distance': self.min_distance_achieved,
            'step_count': self.step_count,
            'collision': False,  # Updated in step()
            'done': False,  # Updated in step()
            'previous_action_taken': self.previous_action_taken
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options (can specify 'initial_state')
            
        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)
        
        # Reset random generator if seed provided
        if seed is not None:
            self.scenario_gen.rng = np.random.RandomState(seed)
        
        # Generate initial scenario
        if options is not None and 'initial_state' in options:
            self.state = np.array(options['initial_state'], dtype=np.float64)
        else:
            if self.scenario_type == 'head_on':
                self.state = self.scenario_gen.generate_head_on_encounter()
            elif self.scenario_type == 'crossing':
                self.state = self.scenario_gen.generate_crossing_encounter()
            else:  # 'random'
                self.state = self.scenario_gen.generate_random_encounter()
        
        # Reset counters
        self.remaining_fuel = self.initial_fuel
        self.step_count = 0
        self.previous_action_taken = False
        self.min_distance_achieved = self.propagator.calculate_distance(self.state)
        self.total_fuel_used = 0.0
        
        # Reset reward function state
        self.reward_config.reset()
        
        # Reset trajectory history
        self.trajectory = [self.state[:2].copy()]
        self.debris_trajectory = [np.array([0.0, 0.0])]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step.
        
        Args:
            action: Action index (0-4)
            
        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")
        
        # Convert action to integer if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        # Map discrete action to control vector
        action_map = {
            0: np.array([0.0, 0.0]),           # Coast
            1: np.array([self.thrust_magnitude, 0.0]),   # +Radial
            2: np.array([-self.thrust_magnitude, 0.0]),  # -Radial
            3: np.array([0.0, self.thrust_magnitude]),   # +Along-track
            4: np.array([0.0, -self.thrust_magnitude])   # -Along-track
        }
        
        control = action_map[action]
        fuel_used = np.linalg.norm(control) * self.dt
        
        # Check fuel availability
        if action != 0 and fuel_used > self.remaining_fuel:
            # Not enough fuel, apply scaled thrust
            scale = self.remaining_fuel / fuel_used
            control = control * scale
            fuel_used = self.remaining_fuel
        
        # Track action
        self.previous_action_taken = (action != 0)
        
        # Update fuel
        self.remaining_fuel -= fuel_used
        self.total_fuel_used += fuel_used
        
        # Propagate dynamics
        self.state = self.propagator.propagate(self.state, control, self.dt)
        self.step_count += 1
        
        # Track minimum distance
        current_distance = self.propagator.calculate_distance(self.state)
        self.min_distance_achieved = min(self.min_distance_achieved, current_distance)
        
        # Store trajectory
        self.trajectory.append(self.state[:2].copy())
        self.debris_trajectory.append(np.array([0.0, 0.0]))
        
        # Check termination conditions
        terminated = False
        truncated = False
        collision = False
        
        # Collision check
        if current_distance < self.collision_radius:
            terminated = True
            collision = True
        
        # Safe pass check (debris is behind and far enough)
        # Behind means y > 0 (debris passed) and distance is safe
        if self.state[1] > 0 and current_distance > self.safe_distance:
            terminated = True
        
        # Fuel depletion
        if self.remaining_fuel <= 0 and not terminated:
            truncated = True
        
        # Max steps
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Get info for reward calculation
        info = self._get_info()
        info['fuel_used'] = fuel_used
        info['collision'] = collision
        info['done'] = terminated or truncated
        
        # Calculate reward
        reward = self.reward_config.calculate_reward(info)
        
        # Get observation
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (if render_mode is set)."""
        if self.render_mode is None:
            return
        
        if self.render_mode == 'human' or self.render_mode == 'rgb_array':
            return self._render_frame()
    
    def _render_frame(self):
        """Render current frame using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except ImportError:
            print("Matplotlib required for rendering")
            return
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot trajectories
        if len(self.trajectory) > 0:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Satellite')
        
        # Plot current positions
        if self.state is not None:
            ax.plot(self.state[0], self.state[1], 'bo', markersize=10, label='Satellite (current)')
        ax.plot(0, 0, 'ro', markersize=10, label='Debris')
        
        # Plot safety zones
        collision_circle = Circle((0, 0), self.collision_radius, 
                                 color='red', fill=False, linestyle='--', 
                                 linewidth=2, label='Collision zone')
        safe_circle = Circle((0, 0), self.safe_distance,
                           color='green', fill=False, linestyle='--',
                           linewidth=2, label='Safe zone')
        ax.add_patch(collision_circle)
        ax.add_patch(safe_circle)
        
        # Formatting
        ax.set_xlabel('Radial Distance (m)', fontsize=12)
        ax.set_ylabel('Along-Track Distance (m)', fontsize=12)
        ax.set_title(f'Orbital Debris Avoidance (Step {self.step_count})', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Set reasonable limits
        max_dist = max(self.safe_distance * 1.5, 
                      max([np.linalg.norm(p) for p in self.trajectory]) if self.trajectory else 1000)
        ax.set_xlim(-max_dist, max_dist)
        ax.set_ylim(-max_dist, max_dist)
        
        if self.render_mode == 'human':
            plt.show()
            plt.close()
        else:  # rgb_array
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
    
    def close(self):
        """Clean up resources."""
        pass


# Register environment with Gymnasium
try:
    from gymnasium.envs.registration import register
    
    register(
        id='OrbitalDebris-v0',
        entry_point='src.environment:OrbitalDebrisEnv',
        max_episode_steps=300,
        reward_threshold=50.0,
    )
except:
    pass  # Registration may fail if module not in path
