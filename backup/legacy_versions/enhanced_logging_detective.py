#!/usr/bin/env python3
"""
Enhanced logging detective agent that tracks name artifacts in detail.

This is a modified version of detective_agent.py with additional logging
to track what happens with each artifact, especially name artifacts.
"""

import logging
import os
import sys
import json
import time
from datetime import datetime
import argparse

# Add the project directory to the path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Set up logging for the diagnostic run
logs_dir = os.path.join(base_dir, "logs", "diagnostic")
os.makedirs(logs_dir, exist_ok=True)

# Configure a special logger just for artifact tracking
artifact_logger = logging.getLogger('artifact_tracker')
artifact_logger.setLevel(logging.DEBUG)

timestamp = int(time.time())
log_file = os.path.join(logs_dir, f'artifact_tracking_{timestamp}.log')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
artifact_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
artifact_logger.addHandler(console_handler)

# Import required modules
from detective_agent import DetectiveAgent
from enhanced_artifact_detector import EnhancedArtifactDetector

# Monkey patch _process_artifacts to add detailed logging
original_process_artifacts = DetectiveAgent._process_artifacts

def enhanced_process_artifacts(self, artifacts, source_url, is_wayback=False, original_url=None):
    """
    Enhanced version of _process_artifacts with detailed logging.
    """
    if not artifacts:
        artifact_logger.info(f"No artifacts found from {source_url}")
        return []
    
    artifact_logger.info(f"Processing {len(artifacts)} artifacts from {source_url}")
    
    # Log all artifacts before processing
    for i, artifact in enumerate(artifacts):
        artifact_type = artifact.get('type', 'unknown')
        artifact_subtype = artifact.get('subtype', 'unknown')
        artifact_content = artifact.get('content', '')
        artifact_name = artifact.get('name', '')
        artifact_score = artifact.get('score', 0)
        
        value = artifact_name if artifact_name else artifact_content
        
        artifact_logger.info(f"ARTIFACT [{i+1}/{len(artifacts)}] - Type: {artifact_type}, Subtype: {artifact_subtype}, "
                            f"Value: '{value}', Score: {artifact_score}")
    
    # Call the original method to get the discoveries
    new_discoveries = original_process_artifacts(self, artifacts, source_url, is_wayback, original_url)
    
    # Log which artifacts became discoveries and which didn't
    discovery_values = [d.get('content', '') for d in new_discoveries]
    
    for i, artifact in enumerate(artifacts):
        artifact_type = artifact.get('type', 'unknown')
        artifact_subtype = artifact.get('subtype', 'unknown')
        artifact_content = artifact.get('content', '')
        artifact_name = artifact.get('name', '')
        artifact_score = artifact.get('score', 0)
        
        value = artifact_name if artifact_name else artifact_content
        
        if any(value in d_value for d_value in discovery_values):
            artifact_logger.info(f"DISCOVERY ADDED ✅ - {artifact_type}/{artifact_subtype} - '{value}' - Score: {artifact_score}")
        else:
            # Check why it wasn't added
            if artifact_score <= 0:
                reason = "Low score (score <= 0)"
            elif self._is_duplicate_discovery({'content': value}):
                reason = "Duplicate discovery"
            else:
                reason = "Unknown reason (possibly type filtering or field mapping issue)"
            
            artifact_logger.info(f"DISCOVERY REJECTED ❌ - {artifact_type}/{artifact_subtype} - '{value}' - "
                               f"Score: {artifact_score} - Reason: {reason}")
    
    return new_discoveries

# Patch the method
DetectiveAgent._process_artifacts = enhanced_process_artifacts

# We also need to patch _is_duplicate_discovery to get more info
original_is_duplicate = DetectiveAgent._is_duplicate_discovery

def enhanced_is_duplicate(self, discovery):
    """Enhanced version of _is_duplicate_discovery with logging."""
    result = original_is_duplicate(self, discovery)
    if result:
        artifact_logger.debug(f"Duplicate check: '{discovery.get('content')}' is a duplicate")
    return result

DetectiveAgent._is_duplicate_discovery = enhanced_is_duplicate

def main():
    """Run the detective agent with enhanced logging."""
    parser = argparse.ArgumentParser(description='Run detective agent with enhanced artifact tracking')
    parser.add_argument('--objective', required=True, help='The research objective')
    parser.add_argument('--entity', required=True, help='The primary entity to investigate')
    parser.add_argument('--max-iterations', type=int, default=1, help='Maximum number of iterations')
    
    args = parser.parse_args()
    
    artifact_logger.info(f"Starting diagnostic run with objective: {args.objective}")
    artifact_logger.info(f"Entity: {args.entity}")
    artifact_logger.info(f"Max iterations: {args.max_iterations}")
    
    # Initialize the detective agent
    agent = DetectiveAgent(
        objective=args.objective,
        entity=args.entity,
        max_iterations=args.max_iterations
    )
    
    # Start the investigation
    discoveries = agent.start_investigation()
    
    # Log final summary
    artifact_logger.info(f"Investigation completed with {len(discoveries)} discoveries")
    
    # Write out a detailed report
    report_path = os.path.join(base_dir, "discrepancy_diagnosis.txt")
    
    with open(report_path, 'w') as f:
        f.write("DISCREPANCY DIAGNOSIS REPORT\n")
        f.write("==========================\n\n")
        f.write(f"Objective: {args.objective}\n")
        f.write(f"Entity: {args.entity}\n")
        f.write(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY\n")
        f.write("-------\n")
        f.write(f"Total discoveries: {len(discoveries)}\n\n")
        
        f.write("DISCOVERIES\n")
        f.write("-----------\n")
        for i, discovery in enumerate(discoveries):
            f.write(f"{i+1}. Type: {discovery.get('type', 'unknown')}\n")
            f.write(f"   Content: {discovery.get('content', 'N/A')}\n")
            f.write(f"   Summary: {discovery.get('summary', 'N/A')}\n")
            f.write(f"   Score: {discovery.get('score', 0)}\n")
            f.write(f"   Source: {discovery.get('source_url', 'N/A')}\n\n")
        
        f.write("\nCONCLUSION\n")
        f.write("----------\n")
        f.write("The detailed artifact tracking log can be found at:\n")
        f.write(f"{log_file}\n\n")
        f.write("Please examine this log to see which artifacts were rejected and why.\n")
    
    artifact_logger.info(f"Detailed diagnosis report written to {report_path}")
    
    return discoveries

if __name__ == "__main__":
    main()