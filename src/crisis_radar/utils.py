"""Shared utility functions for Crisis Radar."""

import logging
import os
from pathlib import Path
from typing import Any, Dict


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if not."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to JSON file."""
    import json
    
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    import json
    
    with open(filepath, "r") as f:
        return json.load(f)

