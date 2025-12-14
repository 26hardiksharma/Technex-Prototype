"""
Custom callbacks and metrics for RL training monitoring
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any


class OrbitalMetricsCallback(BaseCallback):
    """
    Custom callback for logging orbital debris avoidance metrics.
    
    Tracks domain-specific metrics:
    - Collision rate
    - Mean minimum distance
    - Fuel efficiency
    - Success rate
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_collisions = []
        self.episode_min_distances = []
        self.episode_fuel_used = []
        self.episode_successes = []
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        """Called at each step."""
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            # Extract info from environment
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0:
                info = infos[0]
                
                # Record metrics
                collision = info.get('collision', False)
                min_dist = info.get('min_distance', np.inf)
                fuel_used = info.get('total_fuel_used', 0.0)
                distance = info.get('distance', 0.0)
                success = distance > 500.0 and not collision  # Safe pass
                
                self.episode_collisions.append(float(collision))
                self.episode_min_distances.append(min_dist)
                self.episode_fuel_used.append(fuel_used)
                self.episode_successes.append(float(success))
                
                # Log to TensorBoard every N episodes
                if len(self.episode_collisions) >= 10:
                    self.logger.record('orbital/collision_rate', 
                                     np.mean(self.episode_collisions[-100:]))
                    self.logger.record('orbital/mean_min_distance',
                                     np.mean(self.episode_min_distances[-100:]))
                    self.logger.record('orbital/mean_fuel_used',
                                     np.mean(self.episode_fuel_used[-100:]))
                    self.logger.record('orbital/success_rate',
                                     np.mean(self.episode_successes[-100:]))
                    
                    if len(self.episode_min_distances) >= 100:
                        # Statistics over last 100 episodes
                        recent_dists = self.episode_min_distances[-100:]
                        self.logger.record('orbital/min_distance_std',
                                         np.std(recent_dists))
                        self.logger.record('orbital/min_distance_min',
                                         np.min(recent_dists))
        
        return True  # Continue training
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        pass


def calculate_closest_approach(trajectory: np.ndarray) -> float:
    """
    Calculate minimum distance from trajectory.
    
    Args:
        trajectory: Array of shape (n_steps, 4) with [x, y, vx, vy]
        
    Returns:
        Minimum distance achieved
    """
    distances = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 1]**2)
    return np.min(distances)


def calculate_fuel_efficiency(
    fuel_used: float,
    min_distance: float,
    collision: bool
) -> float:
    """
    Calculate fuel efficiency metric.
    
    Higher is better. Accounts for both fuel usage and safety margin.
    
    Args:
        fuel_used: Total fuel consumed (m/s Δv)
        min_distance: Minimum distance achieved (m)
        collision: Whether collision occurred
        
    Returns:
        Efficiency score (higher is better)
    """
    if collision:
        return 0.0
    
    # Reward high safety margin with low fuel use
    # Score = (distance / 500m) / (1 + fuel_used)
    safety_factor = min(min_distance / 500.0, 2.0)  # Cap at 2x
    efficiency = safety_factor / (1.0 + fuel_used)
    
    return efficiency
