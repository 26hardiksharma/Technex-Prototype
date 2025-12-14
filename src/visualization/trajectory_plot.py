"""
Trajectory visualization and analysis tools
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple


def plot_encounter(
    trajectory: np.ndarray,
    collision_radius: float = 75.0,
    safe_distance: float = 500.0,
    title: str = "Orbital Debris Encounter",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a single encounter trajectory.
    
    Args:
        trajectory: Array of shape (n_steps, 4) with [x, y, vx, vy]
        collision_radius: Collision threshold (m)
        safe_distance: Safe separation distance (m)
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Trajectory in position space
    ax1 = axes[0]
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    
    # Plot trajectory with color gradient
    n_points = len(x)
    for i in range(n_points - 1):
        color = plt.cm.viridis(i / n_points)
        ax1.plot(x[i:i+2], y[i:i+2], color=color, linewidth=2)
    
    # Mark start and end
    ax1.plot(x[0], y[0], 'go', markersize=12, label='Start', zorder=5)
    ax1.plot(x[-1], y[-1], 'bs', markersize=12, label='End', zorder=5)
    ax1.plot(0, 0, 'r*', markersize=20, label='Debris', zorder=5)
    
    # Safety zones
    collision_circle = Circle((0, 0), collision_radius, 
                             color='red', fill=False, linestyle='--', 
                             linewidth=2, label='Collision zone', alpha=0.7)
    safe_circle = Circle((0, 0), safe_distance,
                        color='green', fill=False, linestyle='--',
                        linewidth=2, label='Safe zone', alpha=0.7)
    ax1.add_patch(collision_circle)
    ax1.add_patch(safe_circle)
    
    ax1.set_xlabel('Radial Distance (m)', fontsize=12)
    ax1.set_ylabel('Along-Track Distance (m)', fontsize=12)
    ax1.set_title('Position Trajectory', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Right plot: Distance vs Time
    ax2 = axes[1]
    distances = np.sqrt(x**2 + y**2)
    times = np.arange(len(distances)) * 10  # Assuming 10s timestep
    
    ax2.plot(times, distances, 'b-', linewidth=2, label='Distance to debris')
    ax2.axhline(collision_radius, color='red', linestyle='--', 
               linewidth=2, label='Collision radius', alpha=0.7)
    ax2.axhline(safe_distance, color='green', linestyle='--',
               linewidth=2, label='Safe distance', alpha=0.7)
    
    # Mark closest approach
    min_idx = np.argmin(distances)
    ax2.plot(times[min_idx], distances[min_idx], 'ro', 
            markersize=10, label=f'Closest: {distances[min_idx]:.1f}m')
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Distance (m)', fontsize=12)
    ax2.set_title('Distance vs Time', fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_trajectory_comparison(
    trajectories: List[np.ndarray],
    labels: List[str],
    collision_radius: float = 75.0,
    safe_distance: float = 500.0,
    title: str = "Trajectory Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare multiple trajectories on the same plot.
    
    Args:
        trajectories: List of trajectory arrays
        labels: Labels for each trajectory
        collision_radius: Collision threshold (m)
        safe_distance: Safe separation distance (m)
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for traj, label, color in zip(trajectories, labels, colors):
        x = traj[:, 0]
        y = traj[:, 1]
        
        ax.plot(x, y, color=color, linewidth=2, label=label, alpha=0.8)
        ax.plot(x[0], y[0], 'o', color=color, markersize=10)
        ax.plot(x[-1], y[-1], 's', color=color, markersize=10)
    
    # Debris position
    ax.plot(0, 0, 'r*', markersize=25, label='Debris', zorder=10)
    
    # Safety zones
    collision_circle = Circle((0, 0), collision_radius, 
                             color='red', fill=False, linestyle='--', 
                             linewidth=2.5, label='Collision zone', alpha=0.7)
    safe_circle = Circle((0, 0), safe_distance,
                        color='green', fill=False, linestyle='--',
                        linewidth=2.5, label='Safe zone', alpha=0.7)
    ax.add_patch(collision_circle)
    ax.add_patch(safe_circle)
    
    ax.set_xlabel('Radial Distance (m)', fontsize=14)
    ax.set_ylabel('Along-Track Distance (m)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_training_metrics(
    log_dir: str,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training metrics from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        save_path: Path to save figure (optional)
        show: Whether to display plot
        
    Returns:
        Matplotlib figure
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("TensorBoard required for plotting training metrics")
        return None
    
    # Load TensorBoard logs
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get available metrics
    scalar_tags = ea.Tags()['scalars']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('rollout/ep_rew_mean', 'Episode Reward (Mean)'),
        ('orbital/collision_rate', 'Collision Rate'),
        ('orbital/success_rate', 'Success Rate'),
        ('orbital/mean_min_distance', 'Mean Min Distance (m)'),
        ('orbital/mean_fuel_used', 'Mean Fuel Used (m/s)'),
        ('train/learning_rate', 'Learning Rate')
    ]
    
    for idx, (tag, label) in enumerate(metrics_to_plot):
        if tag in scalar_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            axes[idx].plot(steps, values, linewidth=2)
            axes[idx].set_xlabel('Timesteps', fontsize=11)
            axes[idx].set_ylabel(label, fontsize=11)
            axes[idx].set_title(label, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        else:
            axes[idx].text(0.5, 0.5, f'Metric not found:\n{tag}',
                          ha='center', va='center', fontsize=10)
            axes[idx].set_title(label, fontsize=12)
    
    plt.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_animation(
    trajectory: np.ndarray,
    collision_radius: float = 75.0,
    safe_distance: float = 500.0,
    interval: int = 100,
    save_path: Optional[str] = None
) -> FuncAnimation:
    """
    Create animated visualization of encounter.
    
    Args:
        trajectory: Array of shape (n_steps, 4) with [x, y, vx, vy]
        collision_radius: Collision threshold (m)
        safe_distance: Safe separation distance (m)
        interval: Animation interval in milliseconds
        save_path: Path to save animation (optional, .mp4 or .gif)
        
    Returns:
        FuncAnimation object
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Setup plot
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    
    # Initialize plot elements
    trajectory_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.6, label='Trajectory')
    satellite_marker, = ax.plot([], [], 'bo', markersize=12, label='Satellite')
    debris_marker, = ax.plot([0], [0], 'r*', markersize=20, label='Debris')
    
    # Safety zones
    collision_circle = Circle((0, 0), collision_radius, 
                             color='red', fill=False, linestyle='--', 
                             linewidth=2, alpha=0.7)
    safe_circle = Circle((0, 0), safe_distance,
                        color='green', fill=False, linestyle='--',
                        linewidth=2, alpha=0.7)
    ax.add_patch(collision_circle)
    ax.add_patch(safe_circle)
    
    # Format
    ax.set_xlabel('Radial Distance (m)', fontsize=12)
    ax.set_ylabel('Along-Track Distance (m)', fontsize=12)
    ax.set_title('Orbital Debris Encounter Animation', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Set limits
    max_dist = max(safe_distance * 1.5, np.max(np.abs(trajectory[:, :2])))
    ax.set_xlim(-max_dist, max_dist)
    ax.set_ylim(-max_dist, max_dist)
    
    # Animation function
    def animate(frame):
        trajectory_line.set_data(x[:frame+1], y[:frame+1])
        satellite_marker.set_data([x[frame]], [y[frame]])
        
        # Update title with current info
        dist = np.sqrt(x[frame]**2 + y[frame]**2)
        time = frame * 10  # Assuming 10s timestep
        ax.set_title(f'Time: {time}s | Distance: {dist:.1f}m', fontsize=14)
        
        return trajectory_line, satellite_marker
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, frames=len(trajectory),
        interval=interval, blit=True, repeat=True
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=10)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=10)
        print(f"✓ Saved animation to {save_path}")
    
    return anim


def plot_performance_comparison(
    results: dict,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot performance comparison across different agents/methods.
    
    Args:
        results: Dictionary with method names as keys and metrics as values
                 Each value should be a dict with 'collision_rate', 'fuel_used', etc.
        save_path: Path to save figure (optional)
        show: Whether to display plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = list(results.keys())
    metrics = ['collision_rate', 'success_rate', 'mean_fuel_used', 'mean_min_distance']
    titles = ['Collision Rate', 'Success Rate', 'Mean Fuel Used (m/s)', 'Mean Min Distance (m)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        values = [results[m].get(metric, 0) for m in methods]
        bars = ax.bar(methods, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(methods))))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-labels if needed
        if len(max(methods, key=len)) > 10:
            ax.set_xticklabels(methods, rotation=45, ha='right')
    
    plt.suptitle('Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig
