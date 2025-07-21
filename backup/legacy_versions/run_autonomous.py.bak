#!/usr/bin/env python3
"""
Autonomous Runner for the Narrative Discovery Matrix System
----------------------------------------------------------
This script runs the Objectives Manager in autonomous mode with configurable parameters.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from objectives_manager import ObjectivesManager

def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'autonomous_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('autonomous_runner')

def main():
    """Run the Narrative Discovery Matrix System in autonomous mode."""
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description='Run the Narrative Discovery Matrix in autonomous mode')
    parser.add_argument('--cycles', type=int, default=10, 
                        help='Maximum number of objective cycles to run')
    parser.add_argument('--delay', type=int, default=300, 
                        help='Delay in seconds between cycles')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuously until stopped')
    args = parser.parse_args()
    
    logger.info(f"Starting autonomous mode with parameters: cycles={args.cycles}, delay={args.delay}s, continuous={args.continuous}")
    
    print("=" * 80)
    print("Narrative Discovery Matrix System - Autonomous Mode")
    print("=" * 80)
    print(f"Starting with settings:")
    print(f"- Maximum cycles: {'∞' if args.continuous else args.cycles}")
    print(f"- Delay between cycles: {args.delay} seconds")
    print(f"- Log file: {os.path.join('logs', 'autonomous_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log')}")
    print("=" * 80)
    
    try:
        manager = ObjectivesManager()
        
        if args.continuous:
            print("\nRunning in continuous mode. Press Ctrl+C to stop...\n")
            cycle = 1
            while True:
                print(f"\n--- Cycle {cycle} ---")
                manager.run_autonomous_cycle(max_cycles=1, cycle_delay=args.delay)
                cycle += 1
        else:
            manager.run_autonomous_cycle(max_cycles=args.cycles, cycle_delay=args.delay)
            
        print("\nAutonomous operation completed successfully.")
        print(f"Check results/narratives/ directory for narrative-worthy findings.")
        
    except KeyboardInterrupt:
        print("\nAutonomous operation stopped by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during autonomous operation: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()