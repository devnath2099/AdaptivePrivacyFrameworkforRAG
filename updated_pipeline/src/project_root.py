"""Project root utility for robust path resolution across environments.

Uses PROJECT_ROOT env var if set, otherwise resolves from this file's location.
Supports local development, Colab (/content), and Kaggle (/kaggle/working).
"""
from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory.

    Priority:
      1. PROJECT_ROOT environment variable
      2. Parent of this file's directory (i.e., the directory containing src/, configs/, scripts/)
    """
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = get_project_root()

# Common output paths relative to project root
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Ensure src is on the Python path
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))