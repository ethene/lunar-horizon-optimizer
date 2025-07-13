#!/usr/bin/env python3
"""
Lunar Horizon Optimizer - Analysis Examples Runner

This script provides easy commands to reproduce the exact analyses shown in the
progress tracking guide. Use this to test different analysis types and see
progress tracking in action.

Usage:
    python run_analysis_examples.py [quick|standard|production|research]
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n✅ {description} completed successfully!")
        
        # Extract output directory from command
        output_dir = None
        for i, arg in enumerate(cmd):
            if arg == "--output" and i + 1 < len(cmd):
                output_dir = cmd[i + 1]
                break
        
        if output_dir and os.path.exists(output_dir):
            print(f"\n📁 Results saved to: {os.path.abspath(output_dir)}/")
            html_files = [f for f in os.listdir(output_dir) if f.endswith('.html')]
            if html_files:
                print(f"🌐 View visualizations: open {output_dir}/*.html")
                print(f"📊 Available dashboards: {', '.join(html_files)}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  {description} interrupted by user")
        return False

def check_environment():
    """Check if we're in the correct environment."""
    if not os.path.exists("src/cli.py"):
        print("❌ Error: src/cli.py not found. Are you in the Lunar Horizon Optimizer directory?")
        return False
    
    if not os.path.exists("scenarios/01_basic_transfer.json"):
        print("❌ Error: scenarios/01_basic_transfer.json not found.")
        return False
    
    # Check if we're in conda py312 environment
    try:
        result = subprocess.run(["python", "--version"], capture_output=True, text=True)
        if "3.12" not in result.stdout:
            print("⚠️  Warning: You may not be in the py312 conda environment")
            print("   Run: conda activate py312")
    except:
        pass
    
    return True

def quick_analysis():
    """Run quick 30-second analysis."""
    output_dir = f"quick_test_{datetime.now().strftime('%H%M%S')}"
    cmd = [
        "python", "src/cli.py", "analyze",
        "--config", "scenarios/01_basic_transfer.json",
        "--output", output_dir,
        "--population-size", "8",
        "--generations", "5", 
        "--no-sensitivity"
    ]
    return run_command(cmd, "Quick Analysis (30 seconds)")

def standard_analysis():
    """Run standard 1-2 minute analysis."""
    output_dir = f"standard_test_{datetime.now().strftime('%H%M%S')}"
    cmd = [
        "python", "src/cli.py", "analyze",
        "--config", "scenarios/01_basic_transfer.json",
        "--output", output_dir,
        "--population-size", "20",
        "--generations", "15",
        "--no-sensitivity"
    ]
    return run_command(cmd, "Standard Analysis (1-2 minutes)")

def production_analysis():
    """Run production 3-4 minute analysis with sensitivity."""
    output_dir = f"production_test_{datetime.now().strftime('%H%M%S')}"
    cmd = [
        "python", "src/cli.py", "analyze",
        "--config", "scenarios/01_basic_transfer.json",
        "--output", output_dir,
        "--population-size", "52",
        "--generations", "30"
    ]
    return run_command(cmd, "Production Analysis (3-4 minutes, includes sensitivity)")

def research_analysis():
    """Run research-grade 10-30 minute analysis."""
    output_dir = f"research_test_{datetime.now().strftime('%H%M%S')}"
    cmd = [
        "python", "src/cli.py", "analyze",
        "--config", "scenarios/09_complete_mission.json",
        "--output", output_dir,
        "--population-size", "100",
        "--generations", "50"
    ]
    return run_command(cmd, "Research Analysis (10-30 minutes)")

def show_usage():
    """Show usage information."""
    print("""
🚀 Lunar Horizon Optimizer - Analysis Examples

This script reproduces the exact analyses from the Progress Tracking Guide.

Usage:
    python run_analysis_examples.py [analysis_type]

Analysis Types:
    quick       - 30 second validation (8×5, no sensitivity)
    standard    - 1-2 minute analysis (20×15, no sensitivity)  
    production  - 3-4 minute full analysis (52×30, with sensitivity)
    research    - 10-30 minute research grade (100×50)

Examples:
    python run_analysis_examples.py quick
    python run_analysis_examples.py production
    
Expected Results (Apollo-class mission):
    Delta-V: ~22,446 m/s
    NPV: ~$374M
    Total Cost: ~$3,540
    Transfer Time: 4.5 days

Environment Requirements:
    - conda activate py312
    - Run from Lunar Horizon Optimizer root directory
    
For more details, see: PROGRESS_TRACKING_GUIDE.md
""")

def main():
    parser = argparse.ArgumentParser(
        description="Run Lunar Horizon Optimizer analysis examples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "analysis_type", 
        nargs="?",
        choices=["quick", "standard", "production", "research"],
        help="Type of analysis to run"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all analysis types in sequence"
    )
    
    args = parser.parse_args()
    
    if not args.analysis_type and not args.all:
        show_usage()
        return
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    print("🔍 Environment check passed")
    
    # Run analyses
    if args.all:
        print("\n🎯 Running all analysis types in sequence...")
        analyses = [
            ("quick", quick_analysis),
            ("standard", standard_analysis), 
            ("production", production_analysis),
            ("research", research_analysis)
        ]
        
        for name, func in analyses:
            print(f"\n{'🚀' * 3} Starting {name} analysis {'🚀' * 3}")
            success = func()
            if not success:
                print(f"❌ {name} analysis failed, stopping sequence")
                break
    else:
        # Run single analysis
        analyses = {
            "quick": quick_analysis,
            "standard": standard_analysis,
            "production": production_analysis, 
            "research": research_analysis
        }
        
        if args.analysis_type in analyses:
            analyses[args.analysis_type]()
        else:
            print(f"❌ Unknown analysis type: {args.analysis_type}")
            show_usage()

if __name__ == "__main__":
    main()