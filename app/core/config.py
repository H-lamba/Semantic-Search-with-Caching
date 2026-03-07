"""
Centralized configuration loader.
Reads config.yaml and exposes settings as a Python dict.
"""

import os
import yaml

# Go up 3 levels: core/ -> app/ -> project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


def load_config():
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# Singleton config instance
config = load_config()
