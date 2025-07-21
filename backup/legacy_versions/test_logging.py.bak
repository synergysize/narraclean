#!/usr/bin/env python3
"""
Test script for the enhanced logging in name artifact extraction.
"""

import logging
import sys
import os
from enhancements.name_artifact_extractor import NameArtifactExtractor
from detective_agent import DetectiveAgent

# Create a log file handler
log_file = '/tmp/name_extraction_test.log'
if os.path.exists(log_file):
    os.remove(log_file)
    
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add file handler to relevant loggers
for logger_name in ['name_artifact_extractor', 'detective_agent', 'enhanced_artifact_detector']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

def test_name_extraction():
    """Test name extraction with detailed logging."""
    print("Testing name extraction with detailed logging...")
    
    # Sample text with some potential names/usernames
    sample_text = """
    Vitalik Buterin is known for creating Ethereum. He used the username vitalik_buterin on various platforms.
    Before that, he was active on the Bitcoin forum as vbuterin. His early project was called "Ethereum" and
    he collaborated with developers like Gavin Wood (username gavofyork) and others. One of his pseudonyms was
    v_buterin. The Constantinople upgrade was an important milestone in Ethereum's history.
    """
    
    # Create a name artifact extractor
    extractor = NameArtifactExtractor(entity="Vitalik Buterin")
    
    # Extract names
    print("Extracting names from sample text...")
    artifacts = extractor.extract_from_text(sample_text, url="https://example.com/test")
    
    # Print the results
    print(f"\nFound {len(artifacts)} name artifacts:")
    for i, artifact in enumerate(artifacts):
        print(f"{i+1}. {artifact['name']} (type: {artifact['subtype']}, score: {artifact['score']:.2f})")
    
    print("\nSee /tmp/name_extraction_test.log for detailed logging output")

def test_detective_processing():
    """Test detective agent artifact processing with detailed logging."""
    print("\nTesting detective agent artifact processing...")
    
    # Initialize a detective agent
    agent = DetectiveAgent(
        objective="Find name artifacts around Vitalik Buterin",
        entity="Vitalik Buterin",
        max_iterations=1
    )
    
    # Create some test artifacts
    artifacts = [
        {
            'type': 'name_artifact',
            'subtype': 'username',
            'name': 'vbuterin',
            'content': 'vbuterin',
            'summary': 'Username: vbuterin',
            'source_url': 'https://example.com/test',
            'score': 0.8
        },
        {
            'type': 'name_artifact',
            'subtype': 'pseudonym',
            'name': 'v_buterin',
            'content': 'v_buterin',
            'summary': 'Pseudonym: v_buterin',
            'source_url': 'https://example.com/test',
            'score': 0.7
        },
        {
            'type': 'name_artifact',
            'subtype': 'ethereum_upgrades',
            'name': 'Constantinople',
            'content': 'Constantinople',
            'summary': 'Ethereum upgrade: Constantinople',
            'source_url': 'https://example.com/test',
            'score': 0.9
        },
        # Duplicate artifact to test filtering
        {
            'type': 'name_artifact',
            'subtype': 'username',
            'name': 'vbuterin',
            'content': 'vbuterin',
            'summary': 'Username: vbuterin',
            'source_url': 'https://example.com/test2',
            'score': 0.8
        },
        # Entity name to be filtered
        {
            'type': 'name_artifact',
            'subtype': 'name',
            'name': 'Vitalik Buterin',
            'content': 'Vitalik Buterin',
            'summary': 'Name: Vitalik Buterin',
            'source_url': 'https://example.com/test',
            'score': 0.6
        }
    ]
    
    # Process the artifacts
    print("Processing artifacts through detective agent...")
    discoveries = agent._process_artifacts(artifacts, source_url="https://example.com/test")
    
    # Print the results
    print(f"\nProcessed {len(artifacts)} artifacts, got {len(discoveries)} discoveries")
    for i, discovery in enumerate(discoveries):
        print(f"{i+1}. {discovery.get('content')} (type: {discovery.get('type')})")
    
    print("\nSee /tmp/name_extraction_test.log for detailed logging output")

if __name__ == "__main__":
    test_name_extraction()
    test_detective_processing()