"""
Evaluation script for trained PPO agent

This script evaluates a trained model on diverse test scenarios and
generates comprehensive performance reports with visualizations.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.environment.debris_env import OrbitalDebrisEnv
from src.visualization.trajectory_plot import plot_encounter, plot_trajectory_comparison, plot_performance_comparison


def evaluate_agent(
    model_path: str,
    n_episodes: int = 50,
    scenario_types: list = ['random', 'head_on', 'crossing'],
    render: bool = False,
    save_dir: str = './results'
):
    """
    Evaluate trained agent on test scenarios.
    
    Args:
        model_path: Path to trained model
        n_episodes: Number of test episodes per scenario type
        scenario_types: List of scenario types to test
        render: Whether to render episodes
        save_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation results
    """
    print("=" * 80)
    print("EVALUATING TRAINED AGENT")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"Episodes per scenario: {n_episodes}")
    print(f"Scenario types: {scenario_types}")
    
    # Load model
    print("\n[1/4] Loading model...")
    model = PPO.load(model_path)
    print("✓ Model loaded successfully")
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Results storage
    all_results = {}
    
    # Evaluate on each scenario type
    for scenario_type in scenario_types:
        print(f"\n[2/4] Testing on '{scenario_type}' scenarios...")
        
        # Create environment
        env = OrbitalDebrisEnv(
            dt=10.0,
            max_steps=300,
            collision_radius=75.0,
            safe_distance=500.0,
            initial_fuel=10.0,
            thrust_magnitude=0.05,
            reward_type='dense',
            scenario_type=scenario_type,
            render_mode=None
        )
        
        # Run episodes
        episode_results = {
            'rewards': [],
            'collisions': [],
            'min_distances': [],
            'fuel_used': [],
            'successes': [],
            'trajectories': []
        }
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            trajectory = [env.state.copy()]
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                trajectory.append(env.state.copy())
                done = terminated or truncated
                
                if render and ep < 3:  # Render first 3 episodes
                    env.render()
            
            # Record results
            episode_results['rewards'].append(episode_reward)
            episode_results['collisions'].append(float(info['collision']))
            episode_results['min_distances'].append(info['min_distance'])
            episode_results['fuel_used'].append(info['total_fuel_used'])
            
            success = info['distance'] > 500.0 and not info['collision']
            episode_results['successes'].append(float(success))
            episode_results['trajectories'].append(np.array(trajectory))
            
            if (ep + 1) % 10 == 0:
                print(f"  Completed {ep + 1}/{n_episodes} episodes")
        
        env.close()
        
        # Calculate statistics
        results = {
            'mean_reward': np.mean(episode_results['rewards']),
            'std_reward': np.std(episode_results['rewards']),
            'collision_rate': np.mean(episode_results['collisions']),
            'success_rate': np.mean(episode_results['successes']),
            'mean_min_distance': np.mean(episode_results['min_distances']),
            'std_min_distance': np.std(episode_results['min_distances']),
            'mean_fuel_used': np.mean(episode_results['fuel_used']),
            'std_fuel_used': np.std(episode_results['fuel_used']),
            'trajectories': episode_results['trajectories']
        }
        
        all_results[scenario_type] = results
        
        # Print summary
        print(f"\n  Results for '{scenario_type}':")
        print(f"    Success rate: {results['success_rate']*100:.1f}%")
        print(f"    Collision rate: {results['collision_rate']*100:.1f}%")
        print(f"    Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"    Mean min distance: {results['mean_min_distance']:.1f} ± {results['std_min_distance']:.1f} m")
        print(f"    Mean fuel used: {results['mean_fuel_used']:.3f} ± {results['std_fuel_used']:.3f} m/s")
    
    # Generate visualizations
    print("\n[3/4] Generating visualizations...")
    
    # Plot sample trajectories from each scenario
    for scenario_type, results in all_results.items():
        # Plot best and worst trajectories
        distances = [np.min(np.sqrt(t[:, 0]**2 + t[:, 1]**2)) 
                    for t in results['trajectories']]
        
        # Best: largest minimum distance without collision
        safe_trajs = [(d, t) for d, t in zip(distances, results['trajectories']) if d > 75]
        if safe_trajs:
            best_idx = np.argmax([d for d, _ in safe_trajs])
            best_traj = safe_trajs[best_idx][1]
            
            plot_encounter(
                best_traj,
                title=f"Best Avoidance - {scenario_type.title()}",
                save_path=os.path.join(save_dir, f'best_{scenario_type}.png'),
                show=False
            )
    
    # Compare scenarios
    comparison_data = {
        scenario: {
            'collision_rate': results['collision_rate'],
            'success_rate': results['success_rate'],
            'mean_fuel_used': results['mean_fuel_used'],
            'mean_min_distance': results['mean_min_distance']
        }
        for scenario, results in all_results.items()
    }
    
    plot_performance_comparison(
        comparison_data,
        save_path=os.path.join(save_dir, 'scenario_comparison.png'),
        show=False
    )
    
    print(f"✓ Visualizations saved to {save_dir}/")
    
    # Generate report
    print("\n[4/4] Generating evaluation report...")
    
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ORBITAL DEBRIS AVOIDANCE - EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test episodes per scenario: {n_episodes}\n\n")
        
        for scenario_type, results in all_results.items():
            f.write(f"\n{scenario_type.upper()} SCENARIOS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Success rate:      {results['success_rate']*100:6.2f}%\n")
            f.write(f"Collision rate:    {results['collision_rate']*100:6.2f}%\n")
            f.write(f"Mean reward:       {results['mean_reward']:8.2f} ± {results['std_reward']:.2f}\n")
            f.write(f"Mean min distance: {results['mean_min_distance']:8.1f} ± {results['std_min_distance']:.1f} m\n")
            f.write(f"Mean fuel used:    {results['mean_fuel_used']:8.3f} ± {results['std_fuel_used']:.3f} m/s\n")
        
        # Overall statistics
        f.write("\n\nOVERALL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        overall_success = np.mean([r['success_rate'] for r in all_results.values()])
        overall_collision = np.mean([r['collision_rate'] for r in all_results.values()])
        overall_fuel = np.mean([r['mean_fuel_used'] for r in all_results.values()])
        
        f.write(f"Average success rate:   {overall_success*100:6.2f}%\n")
        f.write(f"Average collision rate: {overall_collision*100:6.2f}%\n")
        f.write(f"Average fuel usage:     {overall_fuel:8.3f} m/s\n")
        
        # Safety assessment
        f.write("\n\nSAFETY ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        if overall_collision < 0.05:
            f.write("✓ EXCELLENT: Collision rate < 5%\n")
        elif overall_collision < 0.10:
            f.write("✓ GOOD: Collision rate < 10%\n")
        elif overall_collision < 0.20:
            f.write("⚠ ACCEPTABLE: Collision rate < 20%\n")
        else:
            f.write("✗ NEEDS IMPROVEMENT: Collision rate ≥ 20%\n")
        
        # Fuel efficiency assessment
        f.write("\nFUEL EFFICIENCY ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        if overall_fuel < 1.0:
            f.write("✓ EXCELLENT: Using < 10% of fuel budget\n")
        elif overall_fuel < 3.0:
            f.write("✓ GOOD: Using < 30% of fuel budget\n")
        elif overall_fuel < 5.0:
            f.write("⚠ ACCEPTABLE: Using < 50% of fuel budget\n")
        else:
            f.write("⚠ HIGH FUEL USAGE: Using > 50% of fuel budget\n")
    
    print(f"✓ Report saved to {report_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nOverall Performance:")
    print(f"  Success rate: {overall_success*100:.1f}%")
    print(f"  Collision rate: {overall_collision*100:.1f}%")
    print(f"  Fuel efficiency: {overall_fuel:.3f} m/s average")
    print(f"\nResults saved to: {save_dir}/")
    
    return all_results


def compare_models(
    model_paths: dict,
    n_episodes: int = 50,
    scenario_type: str = 'random',
    save_dir: str = './results/comparison'
):
    """
    Compare multiple trained models.
    
    Args:
        model_paths: Dictionary of {name: path} for models to compare
        n_episodes: Number of test episodes
        scenario_type: Scenario type to test on
        save_dir: Directory to save comparison results
    """
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    os.makedirs(save_dir, exist_ok=True)
    
    comparison_results = {}
    
    for name, path in model_paths.items():
        print(f"\n[{name}] Evaluating...")
        results = evaluate_agent(
            model_path=path,
            n_episodes=n_episodes,
            scenario_types=[scenario_type],
            render=False,
            save_dir=os.path.join(save_dir, name)
        )
        comparison_results[name] = results[scenario_type]
    
    # Generate comparison visualization
    print("\nGenerating comparison plots...")
    
    comparison_data = {
        name: {
            'collision_rate': results['collision_rate'],
            'success_rate': results['success_rate'],
            'mean_fuel_used': results['mean_fuel_used'],
            'mean_min_distance': results['mean_min_distance']
        }
        for name, results in comparison_results.items()
    }
    
    plot_performance_comparison(
        comparison_data,
        save_path=os.path.join(save_dir, 'model_comparison.png'),
        show=False
    )
    
    print(f"✓ Comparison complete! Results in {save_dir}/")
    
    return comparison_results


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate trained debris avoidance agent'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes per scenario (default: 50)')
    parser.add_argument('--scenarios', type=str, nargs='+',
                       default=['random', 'head_on', 'crossing'],
                       help='Scenario types to test')
    parser.add_argument('--render', action='store_true',
                       help='Render test episodes')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    evaluate_agent(
        model_path=args.model,
        n_episodes=args.episodes,
        scenario_types=args.scenarios,
        render=args.render,
        save_dir=args.output
    )


if __name__ == '__main__':
    main()
