#!/opt/anaconda3/envs/py312/bin/python
"""Lunar Horizon Optimizer - Main CLI Entry Point.

This is the main entry point for the Lunar Horizon Optimizer CLI.
It provides both the new modern Click-based interface and backward
compatibility with the original argparse-based interface.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == '__main__':
    # Check if we should use legacy mode
    legacy_mode = (
        len(sys.argv) > 1 and 
        sys.argv[1] in ['analyze', 'config', 'validate', 'sample'] and
        '--legacy' in sys.argv
    )
    
    if legacy_mode:
        # Remove --legacy flag and use original CLI
        sys.argv.remove('--legacy')
        from src.cli import main as legacy_main
        legacy_main()
    else:
        # Use new Click-based CLI
        from src.cli.main import cli
        cli()