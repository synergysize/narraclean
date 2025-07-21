#!/usr/bin/env python3
"""
Artifact Extractor - Extract artifacts from HTML content

This slim version imports core functionality from the modules directory.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import extraction functions from modules
from modules.extractors import (
    extract_artifacts_from_html,
    extract_solidity_contracts,
    extract_wallet_addresses,
    extract_private_keys,
    score_artifact,
    generate_hash
)

def process_file(file_path: str, output_path: str = None):
    """
    Process a file to extract artifacts.
    
    Args:
        file_path: Path to the file to process
        output_path: Path to save the extracted artifacts
    
    Returns:
        List of extracted artifacts
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
        
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Extract artifacts
        artifacts = extract_artifacts_from_html(
            content,
            url=f"file://{os.path.abspath(file_path)}",
            date=datetime.fromtimestamp(os.path.getmtime(file_path))
        )
        
        # Save artifacts if output path provided
        if output_path and artifacts:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(artifacts, f, indent=2)
                
        return artifacts
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return []

def process_url(url: str, output_path: str = None):
    """
    Process a URL to extract artifacts.
    
    Args:
        url: URL to process
        output_path: Path to save the extracted artifacts
    
    Returns:
        List of extracted artifacts
    """
    try:
        import requests
        from datetime import datetime
        
        # Fetch URL content
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract artifacts
        artifacts = extract_artifacts_from_html(
            response.text,
            url=url,
            date=datetime.now()
        )
        
        # Save artifacts if output path provided
        if output_path and artifacts:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(artifacts, f, indent=2)
                
        return artifacts
        
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return []

def main():
    """Main function to run the artifact extractor."""
    parser = argparse.ArgumentParser(description='Extract artifacts from HTML content')
    parser.add_argument('--file', '-f', help='Path to file to process')
    parser.add_argument('--url', '-u', help='URL to process')
    parser.add_argument('--output', '-o', help='Path to save extracted artifacts')
    
    args = parser.parse_args()
    
    if not args.file and not args.url:
        parser.error('Either --file or --url must be provided')
        
    if args.file:
        artifacts = process_file(args.file, args.output)
        print(f"Extracted {len(artifacts)} artifacts from file")
        
    if args.url:
        artifacts = process_url(args.url, args.output)
        print(f"Extracted {len(artifacts)} artifacts from URL")
        
    return 0

if __name__ == '__main__':
    sys.exit(main())