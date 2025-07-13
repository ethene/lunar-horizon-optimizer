#!/usr/bin/env python3
"""Legacy CLI compatibility module.

This module provides backward compatibility with the original argparse-based CLI
while the new Click-based CLI is being developed.
"""

# Import all the original CLI functions for backward compatibility
from src.cli import (
    analyze_command,
    validate_command,
    create_sample_command,
    config_command,
    main as legacy_main,
)

__all__ = [
    "analyze_command",
    "validate_command",
    "create_sample_command",
    "config_command",
    "legacy_main",
]
