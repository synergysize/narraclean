#!/usr/bin/env python3
"""
Simple test script to verify that relative paths are working correctly.
"""

import os
import sys
import logging

# Disable all logging to avoid cluttering the output
logging.disable(logging.CRITICAL)

# Import after disabling logging
from main import base_dir as main_base_dir
from url_queue import URLQueue

def test_paths():
    # Test main.py base_dir
    print(f"main.py base_dir: {main_base_dir}")
    print(f"Expected: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Test url_queue.py state_file
    queue = URLQueue()
    print(f"url_queue.py state_file: {queue.state_file}")
    print(f"Expected: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queue_state.json')}")
    
    # Check if paths match
    main_correct = main_base_dir == os.path.dirname(os.path.abspath(__file__))
    queue_correct = queue.state_file == os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queue_state.json')
    
    if main_correct and queue_correct:
        print("SUCCESS: All paths are correctly using relative paths!")
        return True
    else:
        print("FAILURE: Some paths are not correctly using relative paths.")
        return False

if __name__ == "__main__":
    test_result = test_paths()
    sys.exit(0 if test_result else 1)