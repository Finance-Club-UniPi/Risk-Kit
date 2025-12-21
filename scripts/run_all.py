"""Script to run the complete Crisis Radar pipeline with default settings."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crisis_radar.cli import main

if __name__ == "__main__":
    # Set up arguments for 'run' command with defaults
    sys.argv = ["run_all.py", "run"]
    main()

