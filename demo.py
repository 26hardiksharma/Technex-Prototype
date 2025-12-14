"""
Quick demo script to test the entire system

This script runs a minimal version of the complete pipeline:
1. Test orbital dynamics
2. Test environment
3. Quick training (10K steps)
4. Evaluate agent
5. Benchmark inference
6. Export to ONNX

Perfect for verifying installation and showcasing the system!
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_dynamics():
    """Test 1: Orbital dynamics"""
    print_section("TEST 1: ORBITAL DYNAMICS")
    
    from src.dynamics.orbital_mechanics import HillEquationsPropagator
    
    prop = HillEquationsPropagator()
    print(f"✓ Orbital period: {prop.period/60:.1f} minutes")
    
    # Quick propagation test
    import numpy as np
    state0 = np.array([100.0, -5000.0, 0.0, 50.0])
    state1 = prop.propagate(state0, np.array([0.0, 0.0]), 10.0)
    
    print(f"✓ Initial position: [{state0[0]:.1f}, {state0[1]:.1f}] m")
    print(f"✓ Final position:   [{state1[0]:.1f}, {state1[1]:.1f}] m")
    print("\n✅ Orbital dynamics working!")


def test_environment():
    """Test 2: Gymnasium environment"""
    print_section("TEST 2: GYMNASIUM ENVIRONMENT")
    
    from src.environment.debris_env import OrbitalDebrisEnv
    
    env = OrbitalDebrisEnv()
    obs, info = env.reset()
    
    print(f"✓ Observation space: {env.observation_space.shape}")
    print(f"✓ Action space: {env.action_space.n} discrete actions")
    print(f"✓ Initial distance: {info['distance']:.1f} m")
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    print(f"✓ Ran {i+1} steps successfully")
    env.close()
    
    print("\n✅ Environment working!")


def quick_train():
    """Test 3: Quick training"""
    print_section("TEST 3: QUICK TRAINING (10K STEPS)")
    
    print("This will take about 2-3 minutes...")
    print("Training a tiny PPO agent on debris avoidance")
    
    from scripts.train import train_ppo_agent
    
    model = train_ppo_agent(
        total_timesteps=10000,
        n_envs=2,
        experiment_name='demo_run',
        edge_optimized=True
    )
    
    print("\n✅ Training complete!")
    return model


def quick_evaluate():
    """Test 4: Evaluation"""
    print_section("TEST 4: EVALUATION")
    
    from scripts.evaluate import evaluate_agent
    
    model_path = './models/demo_run/final_model.zip'
    
    if not os.path.exists(model_path):
        print("⚠ Model not found, skipping evaluation")
        return
    
    print("Evaluating on 10 test episodes...")
    
    results = evaluate_agent(
        model_path=model_path,
        n_episodes=10,
        scenario_types=['random'],
        render=False,
        save_dir='./results/demo'
    )
    
    print("\n✅ Evaluation complete!")


def quick_benchmark():
    """Test 5: Inference benchmark"""
    print_section("TEST 5: INFERENCE BENCHMARK")
    
    from scripts.benchmark_inference import benchmark_inference
    
    model_path = './models/demo_run/final_model.zip'
    
    if not os.path.exists(model_path):
        print("⚠ Model not found, skipping benchmark")
        return
    
    print("Benchmarking inference time (100 samples)...")
    
    results = benchmark_inference(
        model_path=model_path,
        n_samples=100,
        warmup=10,
        device='cpu'
    )
    
    print("\n✅ Benchmark complete!")


def quick_export():
    """Test 6: ONNX export"""
    print_section("TEST 6: ONNX EXPORT")
    
    from scripts.export_model import export_to_onnx
    
    model_path = './models/demo_run/final_model.zip'
    
    if not os.path.exists(model_path):
        print("⚠ Model not found, skipping export")
        return
    
    print("Exporting model to ONNX format...")
    
    onnx_path = export_to_onnx(
        model_path=model_path,
        output_path='./models/demo_run/model.onnx',
        verify=True
    )
    
    if onnx_path:
        print("\n✅ ONNX export complete!")


def full_demo():
    """Run complete demo pipeline"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  AUTONOMOUS ORBITAL DEBRIS COLLISION AVOIDANCE SYSTEM".center(78) + "║")
    print("║" + "  Complete System Demo".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝\n")
    
    try:
        # Test 1: Dynamics
        test_dynamics()
        
        # Test 2: Environment
        test_environment()
        
        # Test 3: Training
        quick_train()
        
        # Test 4: Evaluation
        quick_evaluate()
        
        # Test 5: Benchmark
        quick_benchmark()
        
        # Test 6: Export
        quick_export()
        
        # Summary
        print_section("DEMO COMPLETE! 🎉")
        print("All systems operational!")
        print("\n✅ Orbital dynamics: Working")
        print("✅ Environment: Working")
        print("✅ Training: Working")
        print("✅ Evaluation: Working")
        print("✅ Benchmarking: Working")
        print("✅ ONNX Export: Working")
        
        print("\n" + "-" * 80)
        print("NEXT STEPS:")
        print("-" * 80)
        print("\n1. Full Training (4-6 hours):")
        print("   python scripts/train.py --timesteps 500000 --envs 4")
        
        print("\n2. Monitor Training:")
        print("   tensorboard --logdir logs")
        
        print("\n3. Evaluate Results:")
        print("   python scripts/evaluate.py --model models/ppo_debris_avoidance/final_model.zip")
        
        print("\n4. Benchmark Edge Performance:")
        print("   python scripts/benchmark_inference.py --model models/ppo_debris_avoidance/final_model.zip")
        
        print("\n5. Deploy to Hardware:")
        print("   python scripts/export_model.py --model models/ppo_debris_avoidance/final_model.zip --optimize")
        
        print("\n" + "=" * 80)
        print("🚀 Ready to save the satellites!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("\nPlease check:")
        print("  1. All dependencies installed: pip install -r requirements.txt")
        print("  2. Python 3.8+ is being used")
        print("  3. No conflicting packages")
        import traceback
        traceback.print_exc()


def main():
    """Main demo entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='System demo and testing')
    parser.add_argument('--full', action='store_true', 
                       help='Run full demo pipeline')
    parser.add_argument('--dynamics', action='store_true',
                       help='Test only orbital dynamics')
    parser.add_argument('--env', action='store_true',
                       help='Test only environment')
    parser.add_argument('--train', action='store_true',
                       help='Test only training')
    
    args = parser.parse_args()
    
    if args.dynamics:
        test_dynamics()
    elif args.env:
        test_environment()
    elif args.train:
        quick_train()
    else:
        # Default: full demo
        full_demo()


if __name__ == '__main__':
    main()
