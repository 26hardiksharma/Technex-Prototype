"""
Orbital Dynamics using Hill-Clohessy-Wiltshire (HCW) Equations

The HCW equations describe relative motion between two objects in circular orbits,
linearized about the chief (debris) orbit. This is perfect for LEO collision avoidance.

Reference frame: LVLH (Local-Vertical, Local-Horizontal)
- x: radial (away from Earth center)
- y: along-track (direction of motion)
- z: cross-track (out of orbital plane) [ignored in 2D]
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional


class HillEquationsPropagator:
    """Propagate satellite relative motion using Hill-Clohessy-Wiltshire equations."""
    
    def __init__(self, n: float = 0.001027):
        """
        Initialize the propagator.
        
        Args:
            n: Mean motion (rad/s) for circular LEO orbit
               Default: 0.001027 rad/s ≈ 400 km altitude (90-minute orbit)
               n = sqrt(mu / a^3) where mu = 398600 km^3/s^2, a = 6778 km
        """
        self.n = n
        self.period = 2 * np.pi / n  # Orbital period in seconds
        
    def dynamics(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Hill-Clohessy-Wiltshire equations of motion.
        
        State: [x, y, vx, vy] where
            x, y: relative position (m)
            vx, vy: relative velocity (m/s)
        
        Control: [fx, fy] thrust acceleration (m/s^2)
        
        Equations:
            ẍ - 2n·ẏ - 3n²·x = fx
            ÿ + 2n·ẋ = fy
        
        Args:
            t: Time (seconds) - not explicitly used in autonomous system
            state: [x, y, vx, vy] current state
            control: [fx, fy] control acceleration
            
        Returns:
            State derivative [vx, vy, ax, ay]
        """
        x, y, vx, vy = state
        fx, fy = control
        
        # Hill equations
        ax = 2 * self.n * vy + 3 * self.n**2 * x + fx
        ay = -2 * self.n * vx + fy
        
        return np.array([vx, vy, ax, ay])
    
    def propagate(
        self,
        state0: np.ndarray,
        control: np.ndarray,
        dt: float,
        method: str = 'RK45'
    ) -> np.ndarray:
        """
        Propagate state forward by dt with constant control.
        
        Args:
            state0: Initial state [x, y, vx, vy] (m, m/s)
            control: Control acceleration [fx, fy] (m/s^2)
            dt: Time step (seconds)
            method: Integration method ('RK45', 'RK23', 'DOP853')
            
        Returns:
            Final state [x, y, vx, vy]
        """
        # Integrate from t=0 to t=dt
        sol = solve_ivp(
            fun=lambda t, y: self.dynamics(t, y, control),
            t_span=(0, dt),
            y0=state0,
            method=method,
            dense_output=False,
            rtol=1e-8,
            atol=1e-10
        )
        
        return sol.y[:, -1]  # Return final state
    
    def propagate_trajectory(
        self,
        state0: np.ndarray,
        controls: np.ndarray,
        dt: float,
        method: str = 'RK45'
    ) -> np.ndarray:
        """
        Propagate full trajectory with time-varying control.
        
        Args:
            state0: Initial state [x, y, vx, vy]
            controls: Array of controls shape (n_steps, 2)
            dt: Time step (seconds)
            method: Integration method
            
        Returns:
            Trajectory array shape (n_steps+1, 4)
        """
        n_steps = len(controls)
        trajectory = np.zeros((n_steps + 1, 4))
        trajectory[0] = state0
        
        state = state0.copy()
        for i, control in enumerate(controls):
            state = self.propagate(state, control, dt, method)
            trajectory[i + 1] = state
            
        return trajectory
    
    def calculate_distance(self, state: np.ndarray) -> float:
        """Calculate Euclidean distance from origin (debris position)."""
        return np.sqrt(state[0]**2 + state[1]**2)
    
    def calculate_relative_velocity(self, state: np.ndarray) -> float:
        """Calculate relative velocity magnitude."""
        return np.sqrt(state[2]**2 + state[3]**2)
    
    def estimate_time_to_closest_approach(self, state: np.ndarray) -> float:
        """
        Estimate time until closest approach using linear approximation.
        
        If satellite is moving away (positive rate), return large value.
        If moving toward debris, estimate when distance will be minimum.
        
        Args:
            state: Current state [x, y, vx, vy]
            
        Returns:
            Estimated time to closest approach (seconds), or infinity if moving away
        """
        x, y, vx, vy = state
        
        # Distance and its time derivative
        r = np.sqrt(x**2 + y**2)
        if r < 1e-6:  # Already at collision
            return 0.0
        
        # dr/dt = (x·vx + y·vy) / r
        dr_dt = (x * vx + y * vy) / r
        
        if dr_dt >= 0:  # Moving apart
            return np.inf
        
        # Simple linear estimate: time when dr/dt would bring distance to zero
        # More sophisticated: solve quadratic for minimum
        # For now, use -r / dr_dt as first-order approximation
        ttca = -r / dr_dt
        
        # Cap at reasonable value (e.g., half orbit period)
        return min(ttca, self.period / 2)
    
    def calculate_closest_approach_distance(
        self,
        state0: np.ndarray,
        controls: np.ndarray,
        dt: float
    ) -> Tuple[float, int]:
        """
        Calculate minimum distance during trajectory.
        
        Args:
            state0: Initial state
            controls: Control sequence
            dt: Time step
            
        Returns:
            (min_distance, step_index) where minimum occurred
        """
        trajectory = self.propagate_trajectory(state0, controls, dt)
        distances = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 1]**2)
        min_idx = np.argmin(distances)
        return distances[min_idx], min_idx


class ScenarioGenerator:
    """Generate diverse initial conditions for debris encounters."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize random number generator."""
        self.rng = np.random.RandomState(seed)
    
    def generate_head_on_encounter(
        self,
        approach_distance: float = 5000.0,
        approach_velocity: float = 70.0,  # Increased from 50 for more challenge
        offset: float = 50.0  # Reduced offset for tighter encounter
    ) -> np.ndarray:
        """
        Generate head-on collision scenario with moving debris.
        
        Args:
            approach_distance: Initial separation (m)
            approach_velocity: Closure rate (m/s)
            offset: Cross-track offset (m) to allow avoidance
            
        Returns:
            Initial state [x, y, vx, vy]
        """
        # Start far away in along-track direction, nearly on collision course
        x = self.rng.uniform(-offset/2, offset/2)
        y = -approach_distance
        vx = self.rng.uniform(-2, 2)  # Small radial velocity for tighter approach
        vy = approach_velocity  # Approaching along-track
        
        return np.array([x, y, vx, vy])
    
    def generate_crossing_encounter(
        self,
        distance: float = 4000.0,  # Closer starting distance
        relative_velocity: float = 120.0  # Faster debris
    ) -> np.ndarray:
        """
        Generate crossing trajectory scenario with moving debris.
        
        Args:
            distance: Initial separation (m)
            relative_velocity: Relative velocity magnitude (m/s)
            
        Returns:
            Initial state [x, y, vx, vy]
        """
        angle = self.rng.uniform(0, 2 * np.pi)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        
        # Velocity toward origin with tighter cone for more likely collision
        vel_angle = angle + np.pi + self.rng.uniform(-np.pi/6, np.pi/6)
        vx = relative_velocity * np.cos(vel_angle)
        vy = relative_velocity * np.sin(vel_angle)
        
        return np.array([x, y, vx, vy])
    
    def generate_random_encounter(
        self,
        distance_range: Tuple[float, float] = (3000.0, 10000.0),
        velocity_range: Tuple[float, float] = (10.0, 150.0)
    ) -> np.ndarray:
        """
        Generate random encounter with debris on collision course.
        
        Args:
            distance_range: (min, max) initial separation (m)
            velocity_range: (min, max) relative velocity (m/s)
            
        Returns:
            Initial state [x, y, vx, vy]
        """
        # Random position on circle
        distance = self.rng.uniform(*distance_range)
        angle = self.rng.uniform(0, 2 * np.pi)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        
        # Velocity with component toward origin
        vel_mag = self.rng.uniform(*velocity_range)
        # Bias toward collision course (within ±60° of toward-origin direction)
        vel_angle = angle + np.pi + self.rng.uniform(-np.pi/3, np.pi/3)
        vx = vel_mag * np.cos(vel_angle)
        vy = vel_mag * np.sin(vel_angle)
        
        return np.array([x, y, vx, vy])


def test_propagator():
    """Test orbital mechanics implementation."""
    print("Testing Hill Equations Propagator...")
    
    # Initialize propagator for 400 km LEO
    prop = HillEquationsPropagator(n=0.001027)
    print(f"Orbital period: {prop.period/60:.1f} minutes")
    
    # Test 1: Free drift (no control)
    print("\n--- Test 1: Free drift ---")
    state0 = np.array([100.0, -5000.0, 0.0, 50.0])  # Approaching debris
    control = np.array([0.0, 0.0])
    
    distances = []
    times = []
    for step in range(100):
        t = step * 10.0
        state = prop.propagate(state0, control, 10.0 * step)
        dist = prop.calculate_distance(state)
        distances.append(dist)
        times.append(t)
        if step % 20 == 0:
            print(f"  t={t:6.0f}s: distance={dist:7.1f}m, pos=[{state[0]:7.1f}, {state[1]:7.1f}]")
    
    min_dist = min(distances)
    print(f"  Minimum distance: {min_dist:.1f}m at t={times[distances.index(min_dist)]:.0f}s")
    
    # Test 2: Avoidance maneuver
    print("\n--- Test 2: Radial thrust avoidance ---")
    state0 = np.array([50.0, -5000.0, 0.0, 50.0])
    trajectory = []
    
    for step in range(100):
        # Apply radial thrust when close
        dist = prop.calculate_distance(state0)
        if 1000 < dist < 3000:
            control = np.array([0.1, 0.0])  # Radial thrust
        else:
            control = np.array([0.0, 0.0])
        
        state0 = prop.propagate(state0, control, 10.0)
        trajectory.append(state0.copy())
        
        if step % 20 == 0:
            print(f"  t={step*10:6.0f}s: distance={dist:7.1f}m, pos=[{state0[0]:7.1f}, {state0[1]:7.1f}]")
    
    trajectory = np.array(trajectory)
    min_dist_idx = np.argmin(np.sqrt(trajectory[:, 0]**2 + trajectory[:, 1]**2))
    min_dist = np.sqrt(trajectory[min_dist_idx, 0]**2 + trajectory[min_dist_idx, 1]**2)
    print(f"  Minimum distance with avoidance: {min_dist:.1f}m at t={min_dist_idx*10:.0f}s")
    
    print("\n✓ Propagator tests complete!")


if __name__ == "__main__":
    test_propagator()
