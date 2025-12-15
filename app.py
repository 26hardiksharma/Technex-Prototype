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

# Pre-baked demo trajectories so the site can run without a trained model
# These are lightweight, deterministic paths meant for hackathon demos.
STATIC_DEMO = {
    'random': {
        'ai': {
            'trajectory': {
                'satellite': [[0, 0], [40, 60], [90, 115], [145, 165], [205, 205], [260, 230], [305, 245], [340, 250], [360, 248], [370, 240], [375, 230]],
                'original': [[0, 0], [20, 30], [40, 60], [60, 90], [80, 120], [100, 150], [120, 180], [140, 210], [160, 240], [180, 270], [200, 300]],
                'debris': [[0, 0], [-15, -20], [-30, -40], [-45, -60], [-60, -80], [-75, -100], [-90, -120], [-105, -140], [-120, -160], [-135, -180], [-150, -200]],
                'actions': [0, 4, 4, 4, 0, 1, 1, 0, 0, 0],
                'velocities': [[0.2, 0.25], [0.22, 0.27], [0.24, 0.28], [0.25, 0.29], [0.24, 0.28], [0.2, 0.25], [0.18, 0.22], [0.16, 0.2], [0.15, 0.18], [0.14, 0.16], [0.13, 0.15]],
                'time': [i * 10 for i in range(11)],
                'distance': [520, 540, 575, 610, 640, 680, 720, 760, 790, 820, 840],
                'fuel_used': [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.14, 0.16, 0.16, 0.16, 0.16]
            },
            'metrics': {
                'min_distance': 520.0,
                'collision': False,
                'success': True,
                'total_fuel_used': 0.16,
                'episode_reward': 210.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        },
        'no_ai': {
            'trajectory': {
                'satellite': [[0, 0], [15, 22], [30, 44], [45, 66], [60, 88], [75, 110], [90, 132], [105, 154], [120, 176], [135, 198], [150, 220]],
                'time': [i * 10 for i in range(11)],
                'distance': [480, 420, 360, 310, 260, 210, 180, 150, 120, 90, 60]
            },
            'metrics': {
                'min_distance': 60.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -120.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        }
    },
    'head_on': {
        'ai': {
            'trajectory': {
                'satellite': [[-200, 0], [-150, 30], [-90, 70], [-20, 120], [50, 180], [120, 240], [180, 290], [230, 320], [270, 340], [300, 350], [320, 355]],
                'original': [[-200, 0], [-170, 0], [-140, 0], [-110, 0], [-80, 0], [-50, 0], [-20, 0], [10, 0], [40, 0], [70, 0], [100, 0]],
                'debris': [[200, 0], [160, -20], [120, -40], [80, -60], [40, -80], [0, -100], [-40, -120], [-80, -140], [-120, -160], [-160, -180], [-200, -200]],
                'actions': [4, 4, 4, 4, 4, 0, 1, 1, 0, 0],
                'velocities': [[0.4, 0.3], [0.42, 0.32], [0.44, 0.34], [0.46, 0.36], [0.44, 0.34], [0.4, 0.3], [0.32, 0.24], [0.26, 0.2], [0.22, 0.16], [0.2, 0.14], [0.18, 0.12]],
                'time': [i * 10 for i in range(11)],
                'distance': [600, 620, 640, 660, 690, 720, 750, 780, 810, 840, 860],
                'fuel_used': [0.0, 0.03, 0.06, 0.09, 0.12, 0.12, 0.14, 0.16, 0.16, 0.16, 0.16]
            },
            'metrics': {
                'min_distance': 600.0,
                'collision': False,
                'success': True,
                'total_fuel_used': 0.16,
                'episode_reward': 240.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        },
        'no_ai': {
            'trajectory': {
                'satellite': [[-200, 0], [-160, 0], [-120, 0], [-80, 0], [-40, 0], [0, 0], [40, 0], [80, 0], [120, 0], [160, 0], [200, 0]],
                'time': [i * 10 for i in range(11)],
                'distance': [400, 320, 260, 200, 150, 100, 70, 50, 30, 20, 10]
            },
            'metrics': {
                'min_distance': 10.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -180.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        }
    },
    'crossing': {
        'ai': {
            'trajectory': {
                'satellite': [[-150, -150], [-110, -100], [-60, -40], [-10, 20], [40, 80], [90, 130], [140, 170], [180, 200], [210, 220], [230, 230], [240, 235]],
                'original': [[-150, -150], [-130, -130], [-110, -110], [-90, -90], [-70, -70], [-50, -50], [-30, -30], [-10, -10], [10, 10], [30, 30], [50, 50]],
                'debris': [[150, 150], [120, 130], [90, 110], [60, 90], [30, 70], [0, 50], [-30, 30], [-60, 10], [-90, -10], [-120, -30], [-150, -50]],
                'actions': [3, 3, 3, 4, 4, 0, 1, 1, 0, 0],
                'velocities': [[0.35, 0.35], [0.36, 0.36], [0.37, 0.37], [0.38, 0.38], [0.36, 0.36], [0.32, 0.32], [0.28, 0.28], [0.24, 0.24], [0.2, 0.2], [0.18, 0.18], [0.16, 0.16]],
                'time': [i * 10 for i in range(11)],
                'distance': [550, 570, 600, 630, 660, 690, 720, 750, 780, 810, 830],
                'fuel_used': [0.0, 0.02, 0.05, 0.08, 0.1, 0.1, 0.12, 0.14, 0.14, 0.14, 0.14]
            },
            'metrics': {
                'min_distance': 550.0,
                'collision': False,
                'success': True,
                'total_fuel_used': 0.14,
                'episode_reward': 220.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        },
        'no_ai': {
            'trajectory': {
                'satellite': [[-150, -150], [-120, -120], [-90, -90], [-60, -60], [-30, -30], [0, 0], [30, 30], [60, 60], [90, 90], [120, 120], [150, 150]],
                'time': [i * 10 for i in range(11)],
                'distance': [420, 360, 300, 240, 200, 160, 130, 110, 90, 70, 50]
            },
            'metrics': {
                'min_distance': 50.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -150.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        }
    }
}

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


def build_static_run(scenario_type: str, use_ai: bool):
    """Return a canned trajectory/metrics bundle for demo mode."""
    scenario = STATIC_DEMO.get(scenario_type, STATIC_DEMO['random'])
    key = 'ai' if use_ai else 'no_ai'
    selected = scenario[key]
    return {
        'trajectory': selected['trajectory'],
        'metrics': selected['metrics'],
        'scenario_type': scenario_type,
        'use_ai': use_ai
    }


def build_static_compare(scenario_type: str):
    """Return canned AI vs no-AI results for demo mode."""
    scenario = STATIC_DEMO.get(scenario_type, STATIC_DEMO['random'])
    return {
        'ai': scenario['ai'],
        'no_ai': scenario['no_ai']
    }


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
        'ready': True,  # demo mode always runs
        'demo_mode': model is None
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
    data = request.json
    scenario_type = data.get('scenario_type', 'random')
    use_ai = data.get('use_ai', True)
    animate = data.get('animate', False)

    # Demo mode: serve canned data when model is missing
    if model is None:
        return jsonify(build_static_run(scenario_type, use_ai))
    
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
        # Demo mode: fixed benchmark numbers
        return jsonify({
            'mean_latency_ms': 0.85,
            'median_latency_ms': 0.82,
            'min_latency_ms': 0.75,
            'max_latency_ms': 0.95,
            'std_latency_ms': 0.05,
            'p95_latency_ms': 0.92,
            'p99_latency_ms': 0.95,
            'samples': 100,
            'ground_control_latency_ms': 600.0,
            'speedup': 600.0 / 0.85
        })
    
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
    data = request.json
    scenario_type = data.get('scenario_type', 'random')

    # Demo mode
    if model is None:
        return jsonify(build_static_compare(scenario_type))
    
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
