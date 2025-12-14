"""
Flask Web Application for Orbital Debris Avoidance Demo

Interactive web interface to demonstrate the autonomous collision avoidance system.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import json
import time
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory
from stable_baselines3 import PPO

from src.environment.debris_env import OrbitalDebrisEnv
from src.dynamics.orbital_mechanics import ScenarioGenerator

app = Flask(__name__)

# Global variables for model and environment
model = None
env = None
scenario_gen = ScenarioGenerator()

# Load model on startup
MODEL_PATH = './models/demo_run/final_model.zip'


def load_model():
    """Load the trained model."""
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH)
        print("✓ Model loaded successfully")
        return True
    else:
        print(f"⚠ Model not found at {MODEL_PATH}")
        print("Run training first: python scripts/train.py --test")
        return False


def initialize_environment():
    """Initialize the environment."""
    global env
    env = OrbitalDebrisEnv(
        dt=10.0,
        max_steps=300,
        collision_radius=75.0,
        safe_distance=500.0,
        initial_fuel=10.0,
        thrust_magnitude=0.05,
        reward_type='dense',
        scenario_type='random',
        render_mode=None
    )
    print("✓ Environment initialized")


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Check if model is loaded."""
    return jsonify({
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'ready': model is not None
    })


@app.route('/api/run_scenario', methods=['POST'])
def run_scenario():
    """
    Run a collision avoidance scenario.
    
    Request body:
    {
        "scenario_type": "random" | "head_on" | "crossing",
        "use_ai": true | false,
        "animate": true | false  (optional, for real-time animation)
    }
    
    Returns trajectory data and metrics.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    scenario_type = data.get('scenario_type', 'random')
    use_ai = data.get('use_ai', True)
    animate = data.get('animate', False)
    
    # Create environment with specified scenario
    test_env = OrbitalDebrisEnv(
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
    
    # Reset environment
    obs, info = test_env.reset()
    
    # Store initial state for original trajectory calculation
    initial_state = test_env.state.copy()
    
    # Calculate debris trajectory (debris moves in opposite direction from satellite)
    from src.dynamics.orbital_mechanics import HillEquationsPropagator
    propagator = HillEquationsPropagator()
    debris_trajectory = []
    # Debris starts at origin with velocity opposite to satellite's relative velocity
    debris_state = np.array([0.0, 0.0, -initial_state[2], -initial_state[3]])
    for t_step in range(300):
        debris_trajectory.append([float(debris_state[0]), float(debris_state[1])])
        # Debris moves with no control (natural orbital motion)
        debris_state = propagator.propagate(debris_state, control=np.array([0.0, 0.0]), dt=10.0)
    
    # Calculate original trajectory (no AI intervention)
    original_trajectory = []
    original_state = initial_state.copy()
    for t_step in range(300):
        original_trajectory.append([float(original_state[0]), float(original_state[1])])
        # Propagate with no control (coast)
        original_state = propagator.propagate(original_state, control=np.array([0.0, 0.0]), dt=10.0)
    
    # Store trajectory with timesteps for animation
    trajectory = {
        'satellite': [[float(test_env.state[0]), float(test_env.state[1])]],
        'original': [[float(initial_state[0]), float(initial_state[1])]],
        'debris': [[0.0, 0.0]],
        'actions': [],
        'velocities': [[float(test_env.state[2]), float(test_env.state[3])]],
        'time': [0.0],
        'distance': [float(info['distance'])],
        'fuel_used': [0.0],
        'actions': []
    }
    
    episode_reward = 0
    step = 0
    done = False
    
    while not done and step < 300:
        if use_ai:
            # AI decision
            action, _ = model.predict(obs, deterministic=True)
            action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        else:
            # No action (coast)
            action = 0
        
        # Step environment
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_reward += reward
        done = terminated or truncated
        step += 1
        
        # Record trajectory
        trajectory['satellite'].append([float(test_env.state[0]), float(test_env.state[1])])
        # Use the original trajectory point if available, otherwise extend the last point
        if step < len(original_trajectory):
            trajectory['original'].append(original_trajectory[step])
        else:
            trajectory['original'].append(original_trajectory[-1])
        # Add moving debris position
        if step < len(debris_trajectory):
            trajectory['debris'].append(debris_trajectory[step])
        else:
            trajectory['debris'].append(debris_trajectory[-1])
        trajectory['time'].append(float(step * 10))
        trajectory['distance'].append(float(info['distance']))
        trajectory['fuel_used'].append(float(info['total_fuel_used']))
        trajectory['actions'].append(int(action))
    
    test_env.close()
    
    # Calculate results
    min_distance = float(info['min_distance'])
    collision = bool(info['collision'])
    success = min_distance > 500.0 and not collision
    
    return jsonify({
        'trajectory': trajectory,
        'metrics': {
            'min_distance': min_distance,
            'collision': collision,
            'success': success,
            'total_fuel_used': float(info['total_fuel_used']),
            'episode_reward': float(episode_reward),
            'steps': step,
            'duration_seconds': float(step * 10)
        },
        'scenario_type': scenario_type,
        'use_ai': use_ai
    })


@app.route('/api/benchmark', methods=['GET'])
def benchmark():
    """Benchmark inference time."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    n_samples = 100
    latencies = []
    
    # Generate random observations
    obs_shape = (7,)
    
    for _ in range(n_samples):
        obs = np.random.randn(*obs_shape).astype(np.float32)
        
        start = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return jsonify({
        'mean_latency_ms': float(np.mean(latencies)),
        'median_latency_ms': float(np.median(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'samples': n_samples,
        'ground_control_latency_ms': 600.0,
        'speedup': 600.0 / float(np.mean(latencies))
    })


@app.route('/api/compare', methods=['POST'])
def compare():
    """Compare AI vs no-AI (coast) on same scenario."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    scenario_type = data.get('scenario_type', 'random')
    
    # Generate initial state
    if scenario_type == 'head_on':
        initial_state = scenario_gen.generate_head_on_encounter()
    elif scenario_type == 'crossing':
        initial_state = scenario_gen.generate_crossing_encounter()
    else:
        initial_state = scenario_gen.generate_random_encounter()
    
    results = {}
    
    # Run with AI
    for use_ai, label in [(True, 'ai'), (False, 'no_ai')]:
        test_env = OrbitalDebrisEnv(
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
        
        obs, info = test_env.reset(options={'initial_state': initial_state})
        
        trajectory = {
            'satellite': [[float(test_env.state[0]), float(test_env.state[1])]],
            'time': [0.0],
            'distance': [float(info['distance'])]
        }
        
        step = 0
        done = False
        
        while not done and step < 300:
            if use_ai:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
            else:
                action = 0  # Coast
            
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            step += 1
            
            trajectory['satellite'].append([float(test_env.state[0]), float(test_env.state[1])])
            trajectory['time'].append(float(step * 10))
            trajectory['distance'].append(float(info['distance']))
        
        test_env.close()
        
        results[label] = {
            'trajectory': trajectory,
            'min_distance': float(info['min_distance']),
            'collision': bool(info['collision']),
            'fuel_used': float(info['total_fuel_used']),
            'steps': step
        }
    
    return jsonify(results)


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)


if __name__ == '__main__':
    print("=" * 80)
    print("ORBITAL DEBRIS AVOIDANCE - WEB DEMO")
    print("=" * 80)
    
    # Initialize
    model_loaded = load_model()
    initialize_environment()
    
    if not model_loaded:
        print("\n⚠ WARNING: Model not loaded!")
        print("   The app will start but some features won't work.")
        print("   Run: python scripts/train.py --test")
    
    print("\n" + "=" * 80)
    print("Starting Flask server...")
    print("=" * 80)
    print("\n🚀 Open your browser to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
