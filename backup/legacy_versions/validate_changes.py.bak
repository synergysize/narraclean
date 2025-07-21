#!/usr/bin/env python3
"""
Simple validation script for detective agent changes.
"""

import os
import sys

# Check if idle iterations references were removed
def check_idle_iterations_removal():
    with open('detective_agent.py', 'r') as f:
        content = f.read()
    
    # Check if the idle iterations check in _should_continue_investigation was removed
    should_continue_check = "# Removed idle iterations check" in content
    
    # Check if idle iterations increments were removed or commented out
    idle_increments_removed = "self.idle_iterations += 1" not in content
    
    # Check if main loop no longer has max_idle_iterations check
    idle_max_check_removed = "if self.idle_iterations >= self.max_idle_iterations:" not in content
    
    print("\nIdle Iterations Removal Check:")
    print(f"- Idle iterations check in _should_continue_investigation removed: {'✅' if should_continue_check else '❌'}")
    print(f"- Idle iterations increments removed: {'✅' if idle_increments_removed else '❌'}")
    print(f"- Max idle iterations check in main loop removed: {'✅' if idle_max_check_removed else '❌'}")

# Check if fallback URLs were removed
def check_fallback_urls_removal():
    with open('detective_agent.py', 'r') as f:
        content = f.read()
    
    # Check if fallback URLs block was removed
    fallback_block_removed = "adding fallback values for testing" not in content
    
    # Check if proper error message added
    error_added = "LLM strategy generation failed - no targets to investigate" in content
    
    # Check if empty queue check was added
    empty_queue_check = "Research queue is empty after initialization" in content
    
    print("\nFallback URLs Removal Check:")
    print(f"- Fallback URLs block removed: {'✅' if fallback_block_removed else '❌'}")
    print(f"- Proper error message added: {'✅' if error_added else '❌'}")
    print(f"- Empty queue check added: {'✅' if empty_queue_check else '❌'}")

if __name__ == "__main__":
    # Change to the script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Validating detective agent changes...")
    check_idle_iterations_removal()
    check_fallback_urls_removal()