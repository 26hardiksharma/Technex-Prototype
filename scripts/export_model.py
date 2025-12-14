"""
Export trained model to ONNX format for edge deployment

ONNX (Open Neural Network Exchange) is a portable format that can be
deployed on various edge devices including satellite computers.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import onnx
import onnxruntime as ort
from stable_baselines3 import PPO

from src.environment.debris_env import OrbitalDebrisEnv


def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 11,
    verify: bool = True
):
    """
    Export trained PPO model to ONNX format.
    
    Args:
        model_path: Path to trained .zip model
        output_path: Path for output .onnx file
        opset_version: ONNX opset version (11 is widely supported)
        verify: Verify exported model matches original
        
    Returns:
        Path to exported ONNX model
    """
    print("=" * 80)
    print("EXPORTING MODEL TO ONNX FORMAT")
    print("=" * 80)
    print(f"\nInput model: {model_path}")
    print(f"Output ONNX: {output_path}")
    print(f"ONNX opset version: {opset_version}")
    
    # Load model
    print("\n[1/5] Loading trained model...")
    model = PPO.load(model_path)
    print("✓ Model loaded")
    
    # Get observation shape
    env = OrbitalDebrisEnv()
    obs, _ = env.reset()
    obs_shape = obs.shape
    env.close()
    
    print(f"\nObservation shape: {obs_shape}")
    
    # Extract policy network
    print("\n[2/5] Extracting policy network...")
    policy = model.policy
    policy.eval()  # Set to evaluation mode
    
    # Create dummy input
    dummy_input = torch.randn(1, *obs_shape, dtype=torch.float32)
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Network parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024:.2f} KB")
    
    # Export to ONNX
    print("\n[3/5] Exporting to ONNX...")
    
    try:
        torch.onnx.export(
            policy,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,  # Optimize constant folding
            input_names=['observation'],
            output_names=['action_logits', 'value'],
            dynamic_axes={
                'observation': {0: 'batch_size'},
                'action_logits': {0: 'batch_size'},
                'value': {0: 'batch_size'}
            }
        )
        print(f"✓ Export successful: {output_path}")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return None
    
    # Check ONNX model
    print("\n[4/5] Validating ONNX model...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # Print model info
        print(f"\nONNX model info:")
        print(f"  IR version: {onnx_model.ir_version}")
        print(f"  Producer: {onnx_model.producer_name}")
        print(f"  Graph inputs: {len(onnx_model.graph.input)}")
        print(f"  Graph outputs: {len(onnx_model.graph.output)}")
        print(f"  Nodes: {len(onnx_model.graph.node)}")
        
        # File size
        file_size = os.path.getsize(output_path)
        print(f"  File size: {file_size / 1024:.2f} KB")
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return None
    
    # Verify inference matches
    if verify:
        print("\n[5/5] Verifying inference equivalence...")
        
        # PyTorch inference
        with torch.no_grad():
            torch_output = policy(dummy_input)
            torch_action_logits = torch_output[0].numpy()
        
        # ONNX Runtime inference
        ort_session = ort.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_action_logits = ort_outputs[0]
        
        # Compare outputs
        diff = np.abs(torch_action_logits - onnx_action_logits).max()
        print(f"  Max difference: {diff:.2e}")
        
        if diff < 1e-5:
            print("✓ Inference outputs match (diff < 1e-5)")
        elif diff < 1e-3:
            print("⚠ Inference outputs mostly match (diff < 1e-3)")
        else:
            print(f"✗ WARNING: Large difference in outputs ({diff:.2e})")
    
    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)
    print(f"\nONNX model saved to: {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    print("\nDeployment information:")
    print("  - Format: ONNX (Open Neural Network Exchange)")
    print("  - Runtime: ONNX Runtime (CPU, GPU, or embedded)")
    print("  - Targets: x86, ARM, NVIDIA Jetson, Edge TPU, etc.")
    print("  - Optimization: Constant folding enabled")
    
    print("\nNext steps:")
    print("  1. Test with ONNX Runtime: onnxruntime")
    print("  2. Optimize further: onnxruntime-tools")
    print("  3. Quantize for edge: Convert to INT8")
    print("  4. Deploy to target hardware")
    
    return output_path


def benchmark_onnx(onnx_path: str, n_samples: int = 1000):
    """Benchmark ONNX model inference."""
    print("=" * 80)
    print("BENCHMARKING ONNX MODEL")
    print("=" * 80)
    
    # Create session
    print(f"\nLoading ONNX model: {onnx_path}")
    sess = ort.InferenceSession(onnx_path)
    
    # Get input info
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    print(f"Input: {input_name} {input_shape}")
    
    # Benchmark
    print(f"\nBenchmarking {n_samples} inferences...")
    
    import time
    latencies = []
    
    for i in range(n_samples):
        # Random input
        obs = np.random.randn(1, input_shape[1]).astype(np.float32)
        
        # Time inference
        start = time.perf_counter()
        outputs = sess.run(None, {input_name: obs})
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
    
    latencies = np.array(latencies)
    
    print("\n" + "=" * 80)
    print("ONNX BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nMean latency:   {np.mean(latencies):.3f} ms")
    print(f"Median latency: {np.median(latencies):.3f} ms")
    print(f"Min latency:    {np.min(latencies):.3f} ms")
    print(f"Max latency:    {np.max(latencies):.3f} ms")
    print(f"P95 latency:    {np.percentile(latencies, 95):.3f} ms")
    print(f"P99 latency:    {np.percentile(latencies, 99):.3f} ms")
    print(f"\nThroughput:     {1000/np.mean(latencies):.1f} inferences/sec")


def optimize_onnx(onnx_path: str, optimized_path: str):
    """
    Optimize ONNX model for inference.
    
    Applies graph optimizations like constant folding, operator fusion, etc.
    """
    print("=" * 80)
    print("OPTIMIZING ONNX MODEL")
    print("=" * 80)
    
    try:
        from onnxruntime.transformers import optimizer
        
        print(f"\nInput: {onnx_path}")
        print(f"Output: {optimized_path}")
        
        # Load and optimize
        print("\nApplying optimizations...")
        optimized_model = optimizer.optimize_model(
            onnx_path,
            model_type='bert',  # Generic optimization
            num_heads=0,
            hidden_size=0
        )
        
        optimized_model.save_model_to_file(optimized_path)
        
        # Compare sizes
        original_size = os.path.getsize(onnx_path) / 1024
        optimized_size = os.path.getsize(optimized_path) / 1024
        
        print(f"\n✓ Optimization complete")
        print(f"  Original size:  {original_size:.2f} KB")
        print(f"  Optimized size: {optimized_size:.2f} KB")
        print(f"  Reduction:      {original_size - optimized_size:.2f} KB ({(1-optimized_size/original_size)*100:.1f}%)")
        
        return optimized_path
        
    except ImportError:
        print("✗ onnxruntime-tools not installed")
        print("  Install with: pip install onnxruntime-tools")
        return None


def main():
    """Main export entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Export trained model to ONNX format'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for ONNX model (default: same name .onnx)')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version (default: 11)')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification step')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark ONNX model after export')
    parser.add_argument('--optimize', action='store_true',
                       help='Apply additional optimizations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    # Default output path
    if args.output is None:
        args.output = args.model.replace('.zip', '.onnx')
    
    # Export
    onnx_path = export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        opset_version=args.opset,
        verify=not args.no_verify
    )
    
    if onnx_path is None:
        return
    
    # Optimize
    if args.optimize:
        optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        onnx_path = optimize_onnx(onnx_path, optimized_path)
    
    # Benchmark
    if args.benchmark and onnx_path:
        print("\n")
        benchmark_onnx(onnx_path)


if __name__ == '__main__':
    main()
