"""
Benchmark inference time for edge-AI deployment

This script measures the inference latency of the trained neural network
to demonstrate feasibility for onboard satellite deployment.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import torch
from stable_baselines3 import PPO

from src.environment.debris_env import OrbitalDebrisEnv


def benchmark_inference(
    model_path: str,
    n_samples: int = 1000,
    warmup: int = 100,
    batch_size: int = 1,
    device: str = 'cpu'
):
    """
    Benchmark model inference time.
    
    Args:
        model_path: Path to trained model
        n_samples: Number of inference runs
        warmup: Warmup iterations before timing
        batch_size: Batch size for inference
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary of benchmark results
    """
    print("=" * 80)
    print("EDGE-AI INFERENCE BENCHMARKING")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"Device: {device.upper()}")
    print(f"Samples: {n_samples:,}")
    print(f"Warmup iterations: {warmup}")
    print(f"Batch size: {batch_size}")
    
    # Load model
    print("\n[1/5] Loading model...")
    model = PPO.load(model_path, device=device)
    print("✓ Model loaded")
    
    # Create dummy environment for observation space
    env = OrbitalDebrisEnv()
    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    env.close()
    
    print(f"\nObservation dimension: {obs_dim}")
    
    # Count parameters
    print("\n[2/5] Analyzing model architecture...")
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024:.2f} KB (float32)")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")
    
    # Print architecture
    print("\nNetwork architecture:")
    print(model.policy)
    
    # Prepare test data
    print(f"\n[3/5] Preparing test data ({batch_size} samples)...")
    test_obs = np.random.randn(batch_size, obs_dim).astype(np.float32)
    test_obs_tensor = torch.FloatTensor(test_obs).to(device)
    
    # Warmup
    print(f"\n[4/5] Warming up ({warmup} iterations)...")
    model.policy.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.policy(test_obs_tensor)
    print("✓ Warmup complete")
    
    # Benchmark
    print(f"\n[5/5] Benchmarking inference ({n_samples} samples)...")
    
    latencies = []
    
    with torch.no_grad():
        for i in range(n_samples):
            # Generate random observation
            obs = np.random.randn(batch_size, obs_dim).astype(np.float32)
            obs_tensor = torch.FloatTensor(obs).to(device)
            
            # Time inference
            start = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{n_samples}")
    
    latencies = np.array(latencies)
    
    # Calculate statistics
    results = {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'median_latency_ms': np.median(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'throughput_hz': 1000 / np.mean(latencies),
        'total_params': total_params,
        'model_size_kb': total_params * 4 / 1024,
        'device': device
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    print("\nInference Latency:")
    print(f"  Mean:   {results['mean_latency_ms']:8.3f} ms")
    print(f"  Median: {results['median_latency_ms']:8.3f} ms")
    print(f"  Std:    {results['std_latency_ms']:8.3f} ms")
    print(f"  Min:    {results['min_latency_ms']:8.3f} ms")
    print(f"  Max:    {results['max_latency_ms']:8.3f} ms")
    print(f"  P95:    {results['p95_latency_ms']:8.3f} ms")
    print(f"  P99:    {results['p99_latency_ms']:8.3f} ms")
    
    print("\nThroughput:")
    print(f"  {results['throughput_hz']:8.1f} inferences/second")
    print(f"  {results['throughput_hz']*60:8.0f} inferences/minute")
    
    print("\nModel Characteristics:")
    print(f"  Parameters: {results['total_params']:,}")
    print(f"  Model size: {results['model_size_kb']:.2f} KB")
    
    # Real-time capability assessment
    print("\n" + "-" * 80)
    print("REAL-TIME CAPABILITY ASSESSMENT")
    print("-" * 80)
    
    target_freq = 10  # Hz (10 decisions per second)
    target_latency = 1000 / target_freq  # ms
    
    print(f"\nTarget decision frequency: {target_freq} Hz ({target_latency:.1f} ms max latency)")
    print(f"Achieved latency: {results['mean_latency_ms']:.3f} ms")
    
    if results['p99_latency_ms'] < target_latency:
        print(f"\n✓ EXCELLENT: P99 latency ({results['p99_latency_ms']:.3f} ms) < {target_latency:.1f} ms")
        print(f"  System can reliably make decisions at {target_freq} Hz")
    elif results['mean_latency_ms'] < target_latency:
        print(f"\n✓ GOOD: Mean latency ({results['mean_latency_ms']:.3f} ms) < {target_latency:.1f} ms")
        print(f"  System can typically make decisions at {target_freq} Hz")
    else:
        print(f"\n⚠ WARNING: Latency exceeds target!")
        print(f"  Maximum achievable frequency: ~{results['throughput_hz']:.1f} Hz")
    
    # Ground control comparison
    print("\n" + "-" * 80)
    print("GROUND-BASED CONTROL COMPARISON")
    print("-" * 80)
    
    # Typical ground control loop delays
    ground_uplink_ms = 50  # Uplink delay
    ground_downlink_ms = 50  # Downlink delay
    ground_processing_ms = 500  # Ground processing time
    ground_total_ms = ground_uplink_ms + ground_downlink_ms + ground_processing_ms
    
    print(f"\nGround-based control loop:")
    print(f"  Downlink delay:     {ground_downlink_ms:6.0f} ms")
    print(f"  Processing time:    {ground_processing_ms:6.0f} ms")
    print(f"  Uplink delay:       {ground_uplink_ms:6.0f} ms")
    print(f"  Total latency:      {ground_total_ms:6.0f} ms")
    
    speedup = ground_total_ms / results['mean_latency_ms']
    print(f"\nEdge-AI advantage:")
    print(f"  Onboard latency:    {results['mean_latency_ms']:6.3f} ms")
    print(f"  Speedup factor:     {speedup:6.1f}x FASTER")
    print(f"  Time saved:         {ground_total_ms - results['mean_latency_ms']:6.1f} ms per decision")
    
    # Collision avoidance scenario
    print("\n" + "-" * 80)
    print("COLLISION AVOIDANCE SCENARIO")
    print("-" * 80)
    
    closure_velocity = 50  # m/s (typical relative velocity)
    reaction_distance_ground = closure_velocity * (ground_total_ms / 1000)
    reaction_distance_edge = closure_velocity * (results['mean_latency_ms'] / 1000)
    
    print(f"\nAt {closure_velocity} m/s closure velocity:")
    print(f"  Ground control reaction distance: {reaction_distance_ground:6.1f} m")
    print(f"  Edge-AI reaction distance:        {reaction_distance_edge:6.3f} m")
    print(f"  Safety margin improvement:        {reaction_distance_ground - reaction_distance_edge:6.1f} m")
    
    print("\n✓ Edge-AI enables AUTONOMOUS, REAL-TIME collision avoidance")
    print("  Ground control too slow for millisecond decisions!")
    
    return results


def compare_devices(model_path: str, n_samples: int = 1000):
    """Compare inference speed on different devices."""
    print("=" * 80)
    print("MULTI-DEVICE COMPARISON")
    print("=" * 80)
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    results = {}
    
    for device in devices:
        print(f"\n{'='*80}")
        print(f"Testing on {device.upper()}")
        print(f"{'='*80}")
        
        results[device] = benchmark_inference(
            model_path=model_path,
            n_samples=n_samples,
            device=device
        )
    
    # Comparison summary
    if len(devices) > 1:
        print("\n" + "=" * 80)
        print("DEVICE COMPARISON SUMMARY")
        print("=" * 80)
        
        for device in devices:
            print(f"\n{device.upper()}:")
            print(f"  Mean latency: {results[device]['mean_latency_ms']:.3f} ms")
            print(f"  Throughput: {results[device]['throughput_hz']:.1f} Hz")
        
        if 'cuda' in results:
            speedup = results['cpu']['mean_latency_ms'] / results['cuda']['mean_latency_ms']
            print(f"\nGPU Speedup: {speedup:.2f}x faster than CPU")
    
    return results


def profile_model(model_path: str):
    """Profile model to identify bottlenecks."""
    print("=" * 80)
    print("MODEL PROFILING")
    print("=" * 80)
    
    model = PPO.load(model_path)
    
    # Create dummy input
    env = OrbitalDebrisEnv()
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    env.close()
    
    # Profile with PyTorch profiler
    print("\nProfiling forward pass...")
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        with torch.no_grad():
            for _ in range(100):
                model.policy(obs_tensor)
    
    print("\nTop operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    return prof


def main():
    """Main benchmarking entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark edge-AI inference performance'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of inference samples (default: 1000)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to benchmark on (default: cpu)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare CPU and GPU performance')
    parser.add_argument('--profile', action='store_true',
                       help='Profile model to identify bottlenecks')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    if args.compare:
        compare_devices(args.model, args.samples)
    elif args.profile:
        profile_model(args.model)
    else:
        benchmark_inference(
            model_path=args.model,
            n_samples=args.samples,
            device=args.device
        )


if __name__ == '__main__':
    main()
