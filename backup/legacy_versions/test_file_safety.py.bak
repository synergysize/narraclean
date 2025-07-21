#!/usr/bin/env python3
"""
Test the file safety mechanisms.
"""

import logging
# Configure logging to show output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

import os
import sys
import json
import time
import shutil
from url_queue import URLQueue
from narrative_matrix import NarrativeMatrix

def test_url_queue_safety():
    """Test the URL queue's atomic file operations and backup mechanisms."""
    print("=" * 50)
    print("Testing URL Queue file safety...")
    print("=" * 50)
    
    # Initialize the queue with a test state file
    test_state_file = "test_queue_state.json"
    if os.path.exists(test_state_file):
        os.remove(test_state_file)
    if os.path.exists(test_state_file + ".backup"):
        os.remove(test_state_file + ".backup")
    if os.path.exists(test_state_file + ".tmp"):
        os.remove(test_state_file + ".tmp")
    
    print("Creating URL Queue with test state file...")
    queue = URLQueue(state_file=test_state_file)
    
    # Add some URLs
    print("Adding test URLs...")
    queue.add_url("https://example.com")
    queue.add_url("https://test.com")
    
    # Save state first time
    print("Saving queue state first time...")
    save_result = queue.save_state()
    print(f"Save state result (first): {save_result}")
    
    # Verify files were created
    main_exists = os.path.exists(test_state_file)
    backup_exists = os.path.exists(test_state_file + ".backup")
    print(f"After first save:")
    print(f"  - Main state file exists: {main_exists}")
    print(f"  - Backup file exists: {backup_exists}")
    
    # Add another URL and save again to create backup
    print("\nAdding another URL and saving again...")
    queue.add_url("https://example.org")
    save_result = queue.save_state()
    print(f"Save state result (second): {save_result}")
    
    # Verify backup was created on second save
    main_exists = os.path.exists(test_state_file)
    backup_exists = os.path.exists(test_state_file + ".backup")
    print(f"After second save:")
    print(f"  - Main state file exists: {main_exists}")
    print(f"  - Backup file exists: {backup_exists}")
    
    # Examine the file content
    if main_exists:
        with open(test_state_file, 'r') as f:
            content = f.read()
        print(f"Main file size: {len(content)} bytes")
        print("First 100 chars of content:", content[:100])
    
    # Test backup recovery
    print("\nTesting backup recovery...")
    
    # Corrupt the main file
    print("Corrupting main file...")
    with open(test_state_file, 'w') as f:
        f.write("{ This is corrupted JSON }")
    
    # Try to load state
    print("Attempting to load from corrupted file (should use backup)...")
    queue2 = URLQueue(state_file=test_state_file)
    result = queue2.load_state()
    print(f"Load state result after corruption: {result}")
    print(f"Loaded {len(queue2.pending)} pending URLs (should be 2)")
    
    # Verify main file was restored from backup
    main_fixed = False
    try:
        with open(test_state_file, 'r') as f:
            data = json.load(f)
            main_fixed = True
    except json.JSONDecodeError:
        main_fixed = False
    
    print(f"Main file restored from backup: {main_fixed}")
    
    # Clean up
    print("Cleaning up test files...")
    if os.path.exists(test_state_file):
        os.remove(test_state_file)
    if os.path.exists(test_state_file + ".backup"):
        os.remove(test_state_file + ".backup")
    if os.path.exists(test_state_file + ".tmp"):
        os.remove(test_state_file + ".tmp")

def test_narrative_matrix_safety():
    """Test the Narrative Matrix's atomic file operations."""
    print("=" * 50)
    print("Testing Narrative Matrix file safety...")
    print("=" * 50)
    
    # Set up test paths
    test_config_dir = "test_config"
    test_results_dir = "test_results"
    
    # Clean up from previous runs if necessary
    if os.path.exists(test_config_dir):
        shutil.rmtree(test_config_dir)
    if os.path.exists(test_results_dir):
        shutil.rmtree(test_results_dir)
    
    print("Creating test directories...")
    os.makedirs(test_config_dir, exist_ok=True)
    os.makedirs(test_results_dir, exist_ok=True)
    os.makedirs(os.path.join(test_results_dir, "narratives"), exist_ok=True)
    
    test_config_path = os.path.join(test_config_dir, "narrative_matrix.json")
    
    # Initialize matrix with test paths
    print("Initializing Narrative Matrix with test paths...")
    matrix = NarrativeMatrix(config_path=test_config_path)
    matrix.current_objective_path = os.path.join(test_results_dir, "current_objective.txt")
    matrix.narratives_dir = os.path.join(test_results_dir, "narratives")
    
    # Verify config file was created
    config_exists = os.path.exists(test_config_path)
    print(f"Config file exists: {config_exists}")
    
    # Generate an objective
    print("\nGenerating objective...")
    objective = matrix.generate_objective()
    print(f"Generated objective: {objective}")
    
    # Verify objective was saved
    obj_exists = os.path.exists(matrix.current_objective_path)
    print(f"Objective file exists: {obj_exists}")
    
    if obj_exists:
        with open(matrix.current_objective_path, 'r') as f:
            obj_content = f.read()
        print(f"Objective file content: '{obj_content}'")
    
    # Verify no temp files were left
    temp_exists = os.path.exists(matrix.current_objective_path + ".tmp")
    print(f"Temp file exists (should be False): {temp_exists}")
    
    # Test recording discovery
    print("\nTesting discovery recording...")
    matrix.record_discovery({
        "source": "test",
        "content": "Test discovery",
        "entities": ["Test Entity"]
    }, narrative_worthy=True)
    
    # Verify narrative was saved
    narrative_files = os.listdir(matrix.narratives_dir)
    narrative_count = len(narrative_files)
    print(f"Narrative files created: {narrative_count} (should be 1)")
    
    if narrative_count > 0:
        print(f"Narrative file names: {narrative_files}")
        narrative_file = os.path.join(matrix.narratives_dir, narrative_files[0])
        with open(narrative_file, 'r') as f:
            content = f.read()
        print(f"Narrative file size: {len(content)} bytes")
        print("First 100 chars of content:", content[:100])
    
    # Test atomic save for marking objective complete
    print("\nTesting mark_objective_complete...")
    matrix.mark_objective_complete(status="test_completed")
    
    # Verify objective file is gone
    obj_exists = os.path.exists(matrix.current_objective_path)
    print(f"Objective file still exists (should be False): {obj_exists}")
    
    # Clean up
    print("Cleaning up test directories...")
    shutil.rmtree(test_config_dir)
    shutil.rmtree(test_results_dir)

if __name__ == "__main__":
    test_url_queue_safety()
    test_narrative_matrix_safety()
    print("\nAll tests completed!")