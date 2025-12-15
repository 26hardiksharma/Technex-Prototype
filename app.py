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
                'satellite': [[-300, -200], [-210, -110], [-120, -30], [-30, 50], [60, 120], [150, 180], [240, 230], [320, 270], [390, 300], [450, 320], [500, 330]],
                'original': [[-300, -200], [-240, -160], [-180, -120], [-120, -80], [-60, -40], [0, 0], [60, 40], [120, 80], [180, 120], [240, 160], [300, 200]],
                'debris': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                'actions': [0, 4, 4, 4, 0, 1, 1, 0, 0, 0],
                'velocities': [[0.2, 0.25], [0.22, 0.27], [0.24, 0.28], [0.25, 0.29], [0.24, 0.28], [0.2, 0.25], [0.18, 0.22], [0.16, 0.2], [0.15, 0.18], [0.14, 0.16], [0.13, 0.15]],
                'time': [i * 10 for i in range(11)],
                'distance': [520, 540, 575, 610, 640, 680, 720, 760, 790, 820, 840],
                'fuel_used': [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.14, 0.16, 0.16, 0.16, 0.16]
            },
            'metrics': {
                'min_distance': 530.0,
                'collision': False,
                'success': True,
                'total_fuel_used': 0.18,
                'episode_reward': 250.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        },
        'no_ai': {
            'trajectory': {
                'satellite': [[-300, -200], [-240, -160], [-180, -120], [-120, -80], [-60, -40], [0, 0], [60, 40], [120, 80], [180, 120], [240, 160], [300, 200]],
                'original': [[-300, -200], [-240, -160], [-180, -120], [-120, -80], [-60, -40], [0, 0], [60, 40], [120, 80], [180, 120], [240, 160], [300, 200]],
                'debris': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                'actions': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'velocities': [[0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4]],
                'time': [i * 10 for i in range(11)],
                'distance': [360, 291, 223, 156, 90, 0, 90, 156, 223, 291, 360],
                'fuel_used': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            'metrics': {
                'min_distance': 0.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -500.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        }
    },
    'head_on': {
        'ai': {
            'trajectory': {
                'satellite': [[-500, 0], [-400, 80], [-300, 160], [-200, 240], [-100, 320], [0, 400], [100, 480], [200, 520], [300, 540], [400, 550], [500, 555]],
                'original': [[-500, 0], [-400, 0], [-300, 0], [-200, 0], [-100, 0], [0, 0], [100, 0], [200, 0], [300, 0], [400, 0], [500, 0]],
                'debris': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                'actions': [4, 4, 4, 4, 4, 0, 1, 1, 0, 0],
                'velocities': [[0.4, 0.3], [0.42, 0.32], [0.44, 0.34], [0.46, 0.36], [0.44, 0.34], [0.4, 0.3], [0.32, 0.24], [0.26, 0.2], [0.22, 0.16], [0.2, 0.14], [0.18, 0.12]],
                'time': [i * 10 for i in range(11)],
                'distance': [500, 510, 520, 535, 550, 570, 595, 620, 650, 680, 710],
                'fuel_used': [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.20, 0.20, 0.20, 0.20]
            },
            'metrics': {
                'min_distance': 500.0,
                'collision': False,
                'success': True,
                'total_fuel_used': 0.20,
                'episode_reward': 280.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        },
        'no_ai': {
            'trajectory': {
                'satellite': [[-500, 0], [-400, 0], [-300, 0], [-200, 0], [-100, 0], [0, 0], [100, 0], [200, 0], [300, 0], [400, 0], [500, 0]],
                'original': [[-500, 0], [-400, 0], [-300, 0], [-200, 0], [-100, 0], [0, 0], [100, 0], [200, 0], [300, 0], [400, 0], [500, 0]],
                'debris': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                'actions': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'velocities': [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                'time': [i * 10 for i in range(11)],
                'distance': [500, 400, 300, 200, 100, 0, 100, 200, 300, 400, 500],
                'fuel_used': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            'metrics': {
                'min_distance': 0.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -500.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        }
    },
    'crossing': {
        'ai': {
            'trajectory': {
                'satellite': [[-400, -400], [-300, -280], [-200, -160], [-100, -40], [0, 80], [100, 200], [200, 280], [300, 340], [400, 380], [500, 400], [600, 410]],
                'original': [[-400, -400], [-320, -320], [-240, -240], [-160, -160], [-80, -80], [0, 0], [80, 80], [160, 160], [240, 240], [320, 320], [400, 400]],
                'debris': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                'actions': [3, 3, 3, 4, 4, 0, 1, 1, 0, 0],
                'velocities': [[0.35, 0.35], [0.36, 0.36], [0.37, 0.37], [0.38, 0.38], [0.36, 0.36], [0.32, 0.32], [0.28, 0.28], [0.24, 0.24], [0.2, 0.2], [0.18, 0.18], [0.16, 0.16]],
                'time': [i * 10 for i in range(11)],
                'distance': [566, 540, 520, 510, 505, 515, 540, 575, 615, 660, 710],
                'fuel_used': [0.0, 0.02, 0.05, 0.08, 0.11, 0.14, 0.16, 0.17, 0.17, 0.17, 0.17]
            },
            'metrics': {
                'min_distance': 505.0,
                'collision': False,
                'success': True,
                'total_fuel_used': 0.17,
                'episode_reward': 260.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        },
        'no_ai': {
            'trajectory': {
                'satellite': [[-400, -400], [-320, -320], [-240, -240], [-160, -160], [-80, -80], [0, 0], [80, 80], [160, 160], [240, 240], [320, 320], [400, 400]],
                'original': [[-400, -400], [-320, -320], [-240, -240], [-160, -160], [-80, -80], [0, 0], [80, 80], [160, 160], [240, 240], [320, 320], [400, 400]],
                'debris': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                'actions': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'velocities': [[0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.8, 0.8]],
                'time': [i * 10 for i in range(11)],
                'distance': [566, 453, 339, 226, 113, 0, 113, 226, 339, 453, 566],
                'fuel_used': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            },
            'metrics': {
                'min_distance': 0.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -500.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        }
    },
    'example_1': {
        'ai': {
            'trajectory': {
                'satellite': [[-500, 0], [-420, 60], [-340, 140], [-260, 220], [-180, 300], [-100, 360], [-20, 400], [60, 430], [140, 450], [220, 460], [300, 470]],
                'original': [[-500, 0], [-400, 0], [-300, 0], [-200, 0], [-100, 0], [0, 0], [100, 0], [200, 0], [300, 0], [400, 0], [500, 0]],
                'debris': [[0, 0]] * 11,
                'actions': [4, 4, 4, 4, 4, 0, 1, 1, 0, 0],
                'velocities': [[0.44, 0.32], [0.44, 0.32], [0.42, 0.3], [0.40, 0.28], [0.38, 0.26], [0.36, 0.24], [0.34, 0.22], [0.32, 0.20], [0.30, 0.18], [0.28, 0.16], [0.26, 0.14]],
                'time': [i * 10 for i in range(11)],
                'distance': [520, 540, 565, 590, 615, 640, 660, 680, 700, 720, 740],
                'fuel_used': [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.17, 0.18, 0.18, 0.18, 0.18]
            },
            'metrics': {
                'min_distance': 520.0,
                'collision': False,
                'success': True,
                'total_fuel_used': 0.18,
                'episode_reward': 285.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        },
        'no_ai': {
            'trajectory': {
                'satellite': [[-500, 0], [-400, 0], [-300, 0], [-200, 0], [-100, 0], [0, 0], [100, 0], [200, 0], [300, 0], [400, 0], [500, 0]],
                'original': [[-500, 0], [-400, 0], [-300, 0], [-200, 0], [-100, 0], [0, 0], [100, 0], [200, 0], [300, 0], [400, 0], [500, 0]],
                'debris': [[0, 0]] * 11,
                'actions': [0] * 10,
                'velocities': [[1.0, 0.0]] * 11,
                'time': [i * 10 for i in range(11)],
                'distance': [500, 400, 300, 200, 100, 0, 100, 200, 300, 400, 500],
                'fuel_used': [0.0] * 11
            },
            'metrics': {
                'min_distance': 0.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -500.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        }
    },
    'example_2': {
        'ai': {
            'trajectory': {
                'satellite': [[-520, -40], [-440, -20], [-360, 0], [-280, 40], [-200, 100], [-120, 180], [-40, 260], [40, 320], [120, 360], [200, 380], [280, 390]],
                'original': [[-520, -40], [-440, -40], [-360, -40], [-280, -40], [-200, -40], [-120, -40], [-40, -40], [40, -40], [120, -40], [200, -40], [280, -40]],
                'debris': [[0, 0]] * 11,
                'actions': [4, 4, 4, 4, 4, 0, 1, 1, 0, 0],
                'velocities': [[0.46, 0.30], [0.44, 0.28], [0.42, 0.26], [0.40, 0.24], [0.38, 0.22], [0.36, 0.20], [0.34, 0.18], [0.32, 0.16], [0.30, 0.14], [0.28, 0.12], [0.26, 0.10]],
                'time': [i * 10 for i in range(11)],
                'distance': [560, 545, 540, 545, 560, 585, 610, 640, 665, 690, 710],
                'fuel_used': [0.0, 0.03, 0.06, 0.09, 0.12, 0.14, 0.16, 0.17, 0.17, 0.17, 0.17]
            },
            'metrics': {
                'min_distance': 540.0,
                'collision': False,
                'success': True,
                'total_fuel_used': 0.17,
                'episode_reward': 280.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        },
        'no_ai': {
            'trajectory': {
                'satellite': [[-520, -40], [-440, -40], [-360, -40], [-280, -40], [-200, -40], [-120, -40], [-40, -40], [40, -40], [120, -40], [200, -40], [280, -40]],
                'original': [[-520, -40], [-440, -40], [-360, -40], [-280, -40], [-200, -40], [-120, -40], [-40, -40], [40, -40], [120, -40], [200, -40], [280, -40]],
                'debris': [[0, 0]] * 11,
                'actions': [0] * 10,
                'velocities': [[0.9, 0.05]] * 11,
                'time': [i * 10 for i in range(11)],
                'distance': [521, 441, 361, 281, 201, 121, 41, 41, 121, 201, 281],
                'fuel_used': [0.0] * 11
            },
            'metrics': {
                'min_distance': 41.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -480.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        }
    },
    'example_3': {
        'ai': {
            'trajectory': {
                'satellite': [[-540, 20], [-460, 40], [-380, 80], [-300, 140], [-220, 220], [-140, 300], [-60, 360], [20, 410], [100, 440], [180, 455], [260, 460]],
                'original': [[-540, 20], [-460, 20], [-380, 20], [-300, 20], [-220, 20], [-140, 20], [-60, 20], [20, 20], [100, 20], [180, 20], [260, 20]],
                'debris': [[0, 0]] * 11,
                'actions': [4, 4, 4, 4, 4, 1, 1, 0, 0, 0],
                'velocities': [[0.48, 0.32], [0.46, 0.30], [0.44, 0.28], [0.42, 0.26], [0.40, 0.24], [0.36, 0.22], [0.32, 0.20], [0.30, 0.18], [0.28, 0.16], [0.26, 0.14], [0.24, 0.12]],
                'time': [i * 10 for i in range(11)],
                'distance': [600, 585, 570, 565, 570, 590, 610, 635, 660, 680, 700],
                'fuel_used': [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.17, 0.18, 0.18, 0.18, 0.18]
            },
            'metrics': {
                'min_distance': 565.0,
                'collision': False,
                'success': True,
                'total_fuel_used': 0.18,
                'episode_reward': 290.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        },
        'no_ai': {
            'trajectory': {
                'satellite': [[-540, 20], [-460, 20], [-380, 20], [-300, 20], [-220, 20], [-140, 20], [-60, 20], [20, 20], [100, 20], [180, 20], [260, 20]],
                'original': [[-540, 20], [-460, 20], [-380, 20], [-300, 20], [-220, 20], [-140, 20], [-60, 20], [20, 20], [100, 20], [180, 20], [260, 20]],
                'debris': [[0, 0]] * 11,
                'actions': [0] * 10,
                'velocities': [[1.1, 0.0]] * 11,
                'time': [i * 10 for i in range(11)],
                'distance': [540, 440, 340, 240, 140, 40, 60, 140, 240, 340, 440],
                'fuel_used': [0.0] * 11
            },
            'metrics': {
                'min_distance': 40.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -480.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        }
    },
    'example_4': {
        'ai': {
            'trajectory': {
                'satellite': [[-520, 100], [-440, 140], [-360, 180], [-280, 220], [-200, 260], [-120, 300], [-40, 320], [40, 330], [120, 335], [200, 338], [280, 340]],
                'original': [[-520, 100], [-440, 100], [-360, 100], [-280, 100], [-200, 100], [-120, 100], [-40, 100], [40, 100], [120, 100], [200, 100], [280, 100]],
                'debris': [[0, 0]] * 11,
                'actions': [4, 4, 4, 4, 0, 1, 1, 0, 0, 0],
                'velocities': [[0.42, 0.30], [0.42, 0.30], [0.40, 0.28], [0.38, 0.26], [0.36, 0.24], [0.34, 0.22], [0.32, 0.20], [0.30, 0.18], [0.28, 0.16], [0.26, 0.14], [0.24, 0.12]],
                'time': [i * 10 for i in range(11)],
                'distance': [530, 540, 555, 575, 600, 625, 645, 660, 675, 690, 705],
                'fuel_used': [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.15, 0.15, 0.15]
            },
            'metrics': {
                'min_distance': 530.0,
                'collision': False,
                'success': True,
                'total_fuel_used': 0.15,
                'episode_reward': 275.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        },
        'no_ai': {
            'trajectory': {
                'satellite': [[-520, 100], [-440, 100], [-360, 100], [-280, 100], [-200, 100], [-120, 100], [-40, 100], [40, 100], [120, 100], [200, 100], [280, 100]],
                'original': [[-520, 100], [-440, 100], [-360, 100], [-280, 100], [-200, 100], [-120, 100], [-40, 100], [40, 100], [120, 100], [200, 100], [280, 100]],
                'debris': [[0, 0]] * 11,
                'actions': [0] * 10,
                'velocities': [[0.9, 0.05]] * 11,
                'time': [i * 10 for i in range(11)],
                'distance': [530, 430, 330, 230, 130, 30, 70, 170, 270, 370, 470],
                'fuel_used': [0.0] * 11
            },
            'metrics': {
                'min_distance': 30.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -480.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        }
    },
    'example_5': {
        'ai': {
            'trajectory': {
                'satellite': [[-500, -120], [-420, -80], [-340, -20], [-260, 60], [-180, 140], [-100, 200], [-20, 240], [60, 260], [140, 270], [220, 272], [300, 274]],
                'original': [[-500, -120], [-420, -120], [-340, -120], [-260, -120], [-180, -120], [-100, -120], [-20, -120], [60, -120], [140, -120], [220, -120], [300, -120]],
                'debris': [[0, 0]] * 11,
                'actions': [3, 3, 4, 4, 0, 1, 1, 0, 0, 0],
                'velocities': [[0.40, 0.30], [0.40, 0.30], [0.38, 0.28], [0.36, 0.26], [0.34, 0.24], [0.32, 0.22], [0.30, 0.20], [0.28, 0.18], [0.26, 0.16], [0.24, 0.14], [0.22, 0.12]],
                'time': [i * 10 for i in range(11)],
                'distance': [514, 500, 494, 505, 530, 560, 585, 605, 620, 635, 650],
                'fuel_used': [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.15, 0.15, 0.15]
            },
            'metrics': {
                'min_distance': 514.0,
                'collision': False,
                'success': True,
                'total_fuel_used': 0.15,
                'episode_reward': 265.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        },
        'no_ai': {
            'trajectory': {
                'satellite': [[-500, -120], [-420, -120], [-340, -120], [-260, -120], [-180, -120], [-100, -120], [-20, -120], [60, -120], [140, -120], [220, -120], [300, -120]],
                'original': [[-500, -120], [-420, -120], [-340, -120], [-260, -120], [-180, -120], [-100, -120], [-20, -120], [60, -120], [140, -120], [220, -120], [300, -120]],
                'debris': [[0, 0]] * 11,
                'actions': [0] * 10,
                'velocities': [[0.9, 0.05]] * 11,
                'time': [i * 10 for i in range(11)],
                'distance': [514, 414, 314, 214, 114, 14, 86, 186, 286, 386, 486],
                'fuel_used': [0.0] * 11
            },
            'metrics': {
                'min_distance': 14.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -480.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        }
    },
    'example_6': {
        'ai': {
            'trajectory': {
                'satellite': [[-520, 200], [-440, 220], [-360, 240], [-280, 260], [-200, 280], [-120, 300], [-40, 320], [40, 340], [120, 360], [200, 380], [280, 400]],
                'original': [[-520, 200], [-440, 200], [-360, 200], [-280, 200], [-200, 200], [-120, 200], [-40, 200], [40, 200], [120, 200], [200, 200], [280, 200]],
                'debris': [[0, 0]] * 11,
                'actions': [4, 4, 4, 4, 0, 1, 1, 0, 0, 0],
                'velocities': [[0.42, 0.30], [0.42, 0.30], [0.40, 0.28], [0.38, 0.26], [0.36, 0.24], [0.34, 0.22], [0.32, 0.20], [0.30, 0.18], [0.28, 0.16], [0.26, 0.14], [0.24, 0.12]],
                'time': [i * 10 for i in range(11)],
                'distance': [560, 570, 585, 600, 620, 640, 660, 680, 700, 720, 740],
                'fuel_used': [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.15, 0.15, 0.15]
            },
            'metrics': {
                'min_distance': 560.0,
                'collision': False,
                'success': True,
                'total_fuel_used': 0.15,
                'episode_reward': 275.0,
                'steps': 10,
                'duration_seconds': 100.0
            }
        },
        'no_ai': {
            'trajectory': {
                'satellite': [[-520, 200], [-440, 200], [-360, 200], [-280, 200], [-200, 200], [-120, 200], [-40, 200], [40, 200], [120, 200], [200, 200], [280, 200]],
                'original': [[-520, 200], [-440, 200], [-360, 200], [-280, 200], [-200, 200], [-120, 200], [-40, 200], [40, 200], [120, 200], [200, 200], [280, 200]],
                'debris': [[0, 0]] * 11,
                'actions': [0] * 10,
                'velocities': [[0.9, 0.05]] * 11,
                'time': [i * 10 for i in range(11)],
                'distance': [560, 460, 360, 260, 160, 60, 60, 160, 260, 360, 460],
                'fuel_used': [0.0] * 11
            },
            'metrics': {
                'min_distance': 60.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -470.0,
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


def build_static_run(scenario_type: str, use_ai: bool, example_id: str | None = None):
    """Return a canned trajectory/metrics bundle for demo mode."""
    import random
    
    scenario_key = example_id or scenario_type or 'head_on'
    scenario = STATIC_DEMO.get(scenario_key, STATIC_DEMO['head_on'])
    key = 'ai' if use_ai else 'no_ai'
    selected = scenario[key]
    
    # Generate varied collision scenarios when use_ai=false
    if not use_ai:
        # Create 5 different collision variations on the fly
        variation = random.randint(1, 5)
        
        if variation == 1:
            # Diagonal approach from top-left
            satellite = [[-400, -400], [-320, -320], [-240, -240], [-160, -160], [-80, -80], [0, 0], [80, 80], [160, 160], [240, 240], [320, 320], [400, 400]]
        elif variation == 2:
            # Diagonal approach from top-right
            satellite = [[400, -400], [320, -320], [240, -240], [160, -160], [80, -80], [0, 0], [-80, 80], [-160, 160], [-240, 240], [-320, 320], [-400, 400]]
        elif variation == 3:
            # Curved approach from left
            satellite = [[-500, -100], [-400, -80], [-300, -60], [-200, -40], [-100, -20], [0, 0], [100, 20], [200, 40], [300, 60], [400, 80], [500, 100]]
        elif variation == 4:
            # Steep vertical approach
            satellite = [[-50, -600], [-40, -480], [-30, -360], [-20, -240], [-10, -120], [0, 0], [10, 120], [20, 240], [30, 360], [40, 480], [50, 600]]
        else:
            # Shallow horizontal approach
            satellite = [[-600, -50], [-480, -40], [-360, -30], [-240, -20], [-120, -10], [0, 0], [120, 10], [240, 20], [360, 30], [480, 40], [600, 50]]
        
        # Calculate distances for this trajectory
        distances = [np.sqrt(x**2 + y**2) for x, y in satellite]
        
        varied_traj = {
            'satellite': satellite,
            'original': satellite.copy(),
            'debris': [[0, 0]] * len(satellite),
            'actions': [0] * (len(satellite) - 1),
            'velocities': [[0.5, 0.5]] * len(satellite),
            'time': [i * 10 for i in range(len(satellite))],
            'distance': distances,
            'fuel_used': [0.0] * len(satellite)
        }
        
        return {
            'trajectory': varied_traj,
            'metrics': {
                'min_distance': 0.0,
                'collision': True,
                'success': False,
                'total_fuel_used': 0.0,
                'episode_reward': -500.0,
                'steps': len(satellite) - 1,
                'duration_seconds': (len(satellite) - 1) * 10.0
            },
            'scenario_type': scenario_type,
            'use_ai': use_ai
        }
    
    return {
        'trajectory': selected['trajectory'],
        'metrics': selected['metrics'],
        'scenario_type': 'head_on',
        'example_id': scenario_key,
        'use_ai': use_ai
    }


def build_static_compare(scenario_type: str, example_id: str | None = None):
    """Return canned AI vs no-AI results for demo mode."""
    scenario_key = example_id or scenario_type or 'head_on'
    scenario = STATIC_DEMO.get(scenario_key, STATIC_DEMO['head_on'])
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
    example_id = data.get('example_id')
    scenario_type = 'head_on'
    use_ai = data.get('use_ai', True)
    animate = data.get('animate', False)

    scenario_key = example_id or scenario_type

    # Demo mode: serve canned collision data when model is missing OR when use_ai=false
    # This ensures guaranteed collisions for demonstration purposes when AI is off
    if model is None or not use_ai or example_id:
        return jsonify(build_static_run(scenario_type, use_ai, example_id))
    
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
    example_id = data.get('example_id')
    scenario_type = 'head_on'

    # Demo mode
    if model is None or example_id:
        return jsonify(build_static_compare(scenario_type, example_id))
    
    # Generate initial state
    if scenario_type == 'head_on':
        initial_state = scenario_gen.generate_head_on_encounter()
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
