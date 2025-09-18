#!/usr/bin/env python3
"""
Demo script showcasing the enhanced SafetyLit scraper features.
This script demonstrates the key improvements made to the scraping pipeline.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show its output."""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Demonstrate the enhanced features of the SafetyLit scraper."""
    
    print("SafetyLit Scraper Enhancement Demo")
    print("="*50)
    
    # Check if the main script exists
    if not Path("safetylit_pilot.py").exists():
        print("Error: safetylit_pilot.py not found in current directory")
        sys.exit(1)
    
    demos = [
        {
            "cmd": [sys.executable, "safetylit_pilot.py", "--month", "8", "--list"],
            "desc": "List available PDFs (August 2024 - smallest month)"
        },
        {
            "cmd": [sys.executable, "safetylit_pilot.py", "--stats", "--incremental"],
            "desc": "Show processing statistics (requires previous incremental run)"
        },
        {
            "cmd": [sys.executable, "safetylit_pilot.py", "--month", "8", "--incremental", "--quality-control"],
            "desc": "Full enhanced scraping with incremental updates and quality control"
        }
    ]
    
    print("This demo will showcase the following enhancements:")
    print("1. PDF listing functionality")
    print("2. State management and statistics")
    print("3. Incremental updates with quality control")
    print("\nNote: Using August 2024 (month 8) as it has the fewest PDFs for faster demo")
    
    input("\nPress Enter to start the demo...")
    
    for i, demo in enumerate(demos, 1):
        print(f"\n\nDEMO {i}/{len(demos)}")
        success = run_command(demo["cmd"], demo["desc"])
        
        if not success:
            print(f"Demo {i} failed. Check the output above for details.")
            if i < len(demos):
                choice = input("Continue with next demo? (y/n): ").lower().strip()
                if choice != 'y':
                    break
        else:
            print(f"Demo {i} completed successfully!")
        
        if i < len(demos):
            input("\nPress Enter to continue to next demo...")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    
    # Show generated files
    print("\nGenerated files:")
    extensions = ['.csv', '.jsonl', '.json', '.db', '.log']
    for ext in extensions:
        files = list(Path('.').glob(f'*{ext}'))
        if files:
            print(f"\n{ext.upper()} files:")
            for f in files:
                print(f"  - {f}")
    
    print("\nKey enhancements demonstrated:")
    print("✓ Incremental updates - skip already processed content")
    print("✓ Quality control - pandas-based duplicate detection and standardization")
    print("✓ Comprehensive logging - detailed logs and issue tracking")
    print("✓ State management - SQLite database for persistence")
    print("✓ Enhanced reporting - quality metrics and comprehensive analysis")
    
    print(f"\nCheck the README_Enhanced.md for detailed documentation.")

if __name__ == "__main__":
    main()