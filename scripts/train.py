"""
Training script for PPO agent on orbital debris avoidance

This script trains a Proximal Policy Optimization (PPO) agent to autonomously
avoid orbital debris using onboard decision-making.

Key features:
- Small neural network (2-3 layers) for edge deployment
- TensorBoard logging for monitoring
- Checkpoint saving
- Custom metrics tracking
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from src.environment.debris_env import OrbitalDebrisEnv
from src.utils.callbacks import OrbitalMetricsCallback


def create_training_env(n_envs: int = 4, scenario_type: str = 'random'):
    """
    Create vectorized training environments.
    
    Args:
        n_envs: Number of parallel environments
        scenario_type: Type of initial scenarios
        
    Returns:
        Vectorized environment
    """
    env = make_vec_env(
        OrbitalDebrisEnv,
        n_envs=n_envs,
        env_kwargs={
            'dt': 10.0,
            'max_steps': 300,
            'collision_radius': 75.0,
            'safe_distance': 500.0,
            'initial_fuel': 10.0,
            'thrust_magnitude': 0.05,
            'reward_type': 'dense',
            'scenario_type': scenario_type,
            'render_mode': None
        }
    )
    return env


def train_ppo_agent(
    total_timesteps: int = 500000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 10,
    save_dir: str = './models',
    log_dir: str = './logs',
    experiment_name: str = 'ppo_debris_avoidance',
    edge_optimized: bool = True
):
    """
    Train PPO agent for debris avoidance.
    
    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for optimizer
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        save_dir: Directory for saving models
        log_dir: Directory for TensorBoard logs
        experiment_name: Name for this experiment
        edge_optimized: Use small network for edge deployment
    """
    print("=" * 80)
    print("AUTONOMOUS ORBITAL DEBRIS COLLISION AVOIDANCE SYSTEM")
    print("Training PPO Agent with Edge-AI Optimization")
    print("=" * 80)
    print(f"\nExperiment: {experiment_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Edge-optimized network: {edge_optimized}")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create training environment
    print("\n[1/5] Creating training environments...")
    train_env = create_training_env(n_envs=n_envs, scenario_type='random')
    
    # Create evaluation environment
    print("[2/5] Creating evaluation environment...")
    eval_env = create_training_env(n_envs=1, scenario_type='random')
    
    # Configure policy network
    if edge_optimized:
        # Small network for edge deployment (2 layers, 64 neurons each)
        policy_kwargs = dict(
            net_arch=[64, 64],  # ~10K parameters
            activation_fn=torch.nn.ReLU
        )
        print("\n[3/5] Configuring EDGE-OPTIMIZED policy network:")
        print("  Architecture: [64, 64] (2 hidden layers)")
        print("  Activation: ReLU")
        print("  Estimated parameters: ~10,000")
        print("  Target inference time: <10ms")
    else:
        # Standard network (larger)
        policy_kwargs = dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU
        )
        print("\n[3/5] Configuring STANDARD policy network:")
        print("  Architecture: [256, 256] (2 hidden layers)")
        print("  Estimated parameters: ~150,000")
    
    # Create PPO model
    print("\n[4/5] Initializing PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=2048 // n_envs,  # Steps per environment before update
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, experiment_name)
    )
    
    # Print model architecture
    print("\nModel architecture:")
    print(model.policy)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024:.1f} KB (float32)")
    
    # Setup callbacks
    print("\n[5/5] Setting up callbacks...")
    
    # Checkpoint callback: save model every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=os.path.join(save_dir, experiment_name),
        name_prefix='ppo_debris',
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    
    # Evaluation callback: evaluate agent periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, experiment_name, 'best'),
        log_path=os.path.join(log_dir, experiment_name, 'eval'),
        eval_freq=10000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # Custom metrics callback
    metrics_callback = OrbitalMetricsCallback(verbose=1)
    
    # Combine callbacks
    callback = CallbackList([checkpoint_callback, eval_callback, metrics_callback])
    
    print("\nCallbacks configured:")
    print(f"  - Checkpoints every {50000:,} steps")
    print(f"  - Evaluation every {10000:,} steps")
    print(f"  - Custom orbital metrics tracking")
    
    # Start training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print("\nMonitor progress with TensorBoard:")
    print(f"  tensorboard --logdir {log_dir}")
    print("\nTraining metrics:")
    print("  - Episode reward (mean, std)")
    print("  - Collision rate")
    print("  - Mean minimum distance")
    print("  - Fuel efficiency")
    print("  - Success rate")
    print("\n" + "-" * 80)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_model_path = os.path.join(save_dir, experiment_name, 'final_model')
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Close environments
    train_env.close()
    eval_env.close()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModel saved to: {save_dir}/{experiment_name}/")
    print(f"Logs saved to: {log_dir}/{experiment_name}/")
    print("\nNext steps:")
    print("  1. Review training curves in TensorBoard")
    print("  2. Evaluate model: python scripts/evaluate.py")
    print("  3. Benchmark inference: python scripts/benchmark_inference.py")
    print("  4. Export to ONNX: python scripts/export_model.py")
    
    return model


def quick_test():
    """Quick test with minimal training for debugging."""
    print("Running quick test (10K steps)...")
    model = train_ppo_agent(
        total_timesteps=10000,
        n_envs=2,
        experiment_name='test_run',
        edge_optimized=True
    )
    print("\n✓ Quick test completed successfully!")
    return model


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train PPO agent for autonomous debris avoidance'
    )
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps (default: 500000)')
    parser.add_argument('--envs', type=int, default=4,
                       help='Number of parallel environments (default: 4)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--name', type=str, default='ppo_debris_avoidance',
                       help='Experiment name (default: ppo_debris_avoidance)')
    parser.add_argument('--edge', action='store_true', default=True,
                       help='Use edge-optimized small network (default: True)')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test (10K steps)')
    
    args = parser.parse_args()
    
    if args.test:
        quick_test()
    else:
        train_ppo_agent(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            learning_rate=args.lr,
            experiment_name=args.name,
            edge_optimized=args.edge
        )


if __name__ == '__main__':
    main()
