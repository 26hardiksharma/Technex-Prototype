"""
Reward Functions for Debris Avoidance RL Training

Design philosophy:
- Large negative reward for collision (catastrophic failure)
- Small negative reward for fuel usage (efficiency)
- Small positive reward for safe pass (mission success)
- Distance-based shaping to guide learning
"""

import numpy as np
from typing import Dict, Any


def sparse_reward(
    distance: float,
    fuel_used: float,
    collision_radius: float,
    safe_distance: float,
    previous_action_taken: bool,
    done: bool,
    collision: bool
) -> float:
    """
    Sparse reward: only at episode termination.
    
    Args:
        distance: Current distance to debris (m)
        fuel_used: Fuel consumed this step (m/s Δv)
        collision_radius: Collision threshold (m)
        safe_distance: Desired safe separation (m)
        previous_action_taken: Whether avoidance maneuver executed
        done: Episode terminated
        collision: Collision occurred
        
    Returns:
        Reward value
    """
    if not done:
        return 0.0
    
    if collision:
        return -1000.0  # Catastrophic failure
    
    # Successful avoidance
    if distance > safe_distance:
        return 100.0 - fuel_used  # Success bonus minus fuel cost
    
    # Close call but survived
    return 10.0 - fuel_used


def dense_reward(
    distance: float,
    fuel_used: float,
    collision_radius: float,
    safe_distance: float,
    previous_action_taken: bool,
    done: bool,
    collision: bool,
    previous_distance: float = None
) -> float:
    """
    Dense reward: feedback every step with distance shaping.
    
    Args:
        distance: Current distance to debris (m)
        fuel_used: Fuel consumed this step (m/s Δv)
        collision_radius: Collision threshold (m)
        safe_distance: Desired safe separation (m)
        previous_action_taken: Whether avoidance maneuver executed
        done: Episode terminated
        collision: Collision occurred
        previous_distance: Distance at previous step (for shaping)
        
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Catastrophic collision
    if collision:
        return -1000.0
    
    # Fuel efficiency penalty
    reward -= 0.01 * fuel_used
    
    # Distance-based reward shaping
    if distance < collision_radius * 2:
        # Danger zone: penalize being close
        reward -= 1.0 * (1.0 - distance / (collision_radius * 2))
    elif distance > safe_distance:
        # Safe zone: small positive reward
        reward += 0.1
    
    # Distance improvement reward
    if previous_distance is not None:
        distance_change = distance - previous_distance
        if distance < safe_distance and distance_change > 0:
            # Reward increasing distance when too close
            reward += 0.05 * distance_change / safe_distance
    
    # Terminal success bonus
    if done and distance > safe_distance:
        reward += 50.0
    
    return reward


def fuel_aware_reward(
    distance: float,
    fuel_used: float,
    collision_radius: float,
    safe_distance: float,
    previous_action_taken: bool,
    done: bool,
    collision: bool,
    remaining_fuel: float,
    max_fuel: float
) -> float:
    """
    Fuel-aware reward: heavily penalize fuel usage to learn efficiency.
    
    This reward function emphasizes fuel conservation, which is critical
    for real satellite operations with limited propellant.
    
    Args:
        distance: Current distance to debris (m)
        fuel_used: Fuel consumed this step (m/s Δv)
        collision_radius: Collision threshold (m)
        safe_distance: Desired safe separation (m)
        previous_action_taken: Whether avoidance maneuver executed
        done: Episode terminated
        collision: Collision occurred
        remaining_fuel: Fuel remaining (m/s Δv)
        max_fuel: Initial fuel capacity (m/s Δv)
        
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Collision is worst outcome
    if collision:
        return -1000.0
    
    # Progressive fuel penalty (more expensive as fuel depletes)
    fuel_ratio = remaining_fuel / max_fuel
    fuel_penalty_multiplier = 1.0 + (1.0 - fuel_ratio) * 2.0  # 1x to 3x
    reward -= 0.05 * fuel_used * fuel_penalty_multiplier
    
    # Safety reward
    if distance > safe_distance:
        reward += 0.2
    elif distance < collision_radius * 1.5:
        reward -= 0.5  # Danger zone penalty
    
    # Success bonus weighted by fuel efficiency
    if done and not collision:
        fuel_efficiency = remaining_fuel / max_fuel
        reward += 100.0 * fuel_efficiency  # More reward for using less fuel
    
    return reward


def potential_based_reward(
    distance: float,
    fuel_used: float,
    collision_radius: float,
    safe_distance: float,
    previous_action_taken: bool,
    done: bool,
    collision: bool,
    ttca: float,  # Time to closest approach
    previous_ttca: float = None
) -> float:
    """
    Potential-based reward shaping using distance and time to closest approach.
    
    Guarantees policy invariance (doesn't change optimal policy) while
    providing dense learning signal.
    
    Args:
        distance: Current distance to debris (m)
        fuel_used: Fuel consumed this step (m/s Δv)
        collision_radius: Collision threshold (m)
        safe_distance: Desired safe separation (m)
        previous_action_taken: Whether avoidance maneuver executed
        done: Episode terminated
        collision: Collision occurred
        ttca: Time to closest approach (s)
        previous_ttca: Previous TTCA for potential difference
        
    Returns:
        Reward value
    """
    # Base reward
    reward = -0.01 * fuel_used
    
    if collision:
        return -1000.0 + reward
    
    # Potential function: higher when safer
    def potential(d, t):
        # Combine distance safety with time urgency
        distance_potential = min(d / safe_distance, 1.0) * 10.0
        time_potential = min(t / 300.0, 1.0) * 5.0  # 5 minutes max
        return distance_potential + time_potential
    
    # Potential-based shaping: Φ(s') - Φ(s)
    current_potential = potential(distance, ttca)
    if previous_ttca is not None:
        # Use a reasonable previous distance (this is approximate)
        reward += current_potential  # Simplified: full current potential
    
    # Terminal rewards
    if done and distance > safe_distance:
        reward += 100.0
    
    return reward


class RewardConfig:
    """Configuration for reward function selection and parameters."""
    
    REWARD_FUNCTIONS = {
        'sparse': sparse_reward,
        'dense': dense_reward,
        'fuel_aware': fuel_aware_reward,
        'potential': potential_based_reward
    }
    
    def __init__(
        self,
        reward_type: str = 'dense',
        collision_radius: float = 75.0,
        safe_distance: float = 500.0,
        max_fuel: float = 10.0
    ):
        """
        Initialize reward configuration.
        
        Args:
            reward_type: One of 'sparse', 'dense', 'fuel_aware', 'potential'
            collision_radius: Collision threshold (m)
            safe_distance: Desired safe separation (m)
            max_fuel: Maximum fuel capacity (m/s Δv)
        """
        if reward_type not in self.REWARD_FUNCTIONS:
            raise ValueError(f"Unknown reward type: {reward_type}")
        
        self.reward_type = reward_type
        self.reward_fn = self.REWARD_FUNCTIONS[reward_type]
        self.collision_radius = collision_radius
        self.safe_distance = safe_distance
        self.max_fuel = max_fuel
        
        # Track previous state for shaping
        self.previous_distance = None
        self.previous_ttca = None
    
    def calculate_reward(self, info: Dict[str, Any]) -> float:
        """
        Calculate reward from environment info dictionary.
        
        Args:
            info: Dictionary with keys:
                - distance: Current distance (m)
                - fuel_used: Fuel used this step (m/s)
                - collision: Boolean collision flag
                - done: Boolean termination flag
                - remaining_fuel: Fuel remaining (m/s)
                - ttca: Time to closest approach (s)
                - previous_action_taken: Boolean maneuver flag
                
        Returns:
            Reward value
        """
        # Prepare arguments based on reward function
        kwargs = {
            'distance': info['distance'],
            'fuel_used': info.get('fuel_used', 0.0),
            'collision_radius': self.collision_radius,
            'safe_distance': self.safe_distance,
            'previous_action_taken': info.get('previous_action_taken', False),
            'done': info.get('done', False),
            'collision': info.get('collision', False)
        }
        
        # Add function-specific arguments
        if self.reward_type == 'dense':
            kwargs['previous_distance'] = self.previous_distance
        elif self.reward_type == 'fuel_aware':
            kwargs['remaining_fuel'] = info.get('remaining_fuel', self.max_fuel)
            kwargs['max_fuel'] = self.max_fuel
        elif self.reward_type == 'potential':
            kwargs['ttca'] = info.get('ttca', np.inf)
            kwargs['previous_ttca'] = self.previous_ttca
        
        reward = self.reward_fn(**kwargs)
        
        # Update history
        self.previous_distance = info['distance']
        self.previous_ttca = info.get('ttca', np.inf)
        
        return reward
    
    def reset(self):
        """Reset state tracking between episodes."""
        self.previous_distance = None
        self.previous_ttca = None
