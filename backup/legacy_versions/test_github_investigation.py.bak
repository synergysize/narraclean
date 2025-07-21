#!/usr/bin/env python3
"""
Test script for the GitHub investigation functionality.
"""

import logging
import sys
from detective_agent import DetectiveAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('test_github')

def test_github_investigation():
    """Test the GitHub investigation functionality."""
    # Create a detective agent instance
    detective = DetectiveAgent(
        objective="Find name artifacts in Ethereum repositories",
        entity="Vitalik Buterin",
        max_iterations=5,
        max_time_hours=1,
        max_idle_iterations=2
    )
    
    # Test repositories to investigate
    test_repos = [
        {'url': 'https://github.com/ethereum/web3.py', 'type': 'github'},
        {'url': 'https://github.com/ethereum/solidity', 'type': 'github'},
        {'url': 'https://github.com/vbuterin/pybitcointools', 'type': 'github'}
    ]
    
    all_discoveries = []
    
    # Test each repository
    for repo in test_repos:
        logger.info(f"Testing GitHub investigation for: {repo['url']}")
        
        try:
            # Extract the owner and repo for logging
            owner, repo_name = detective._extract_github_repo_info(repo['url'])
            logger.info(f"Extracted repository info: {owner}/{repo_name}")
            
            # Run the investigation
            discoveries = detective._investigate_github(repo)
            
            logger.info(f"Found {len(discoveries)} artifacts in {repo['url']}")
            all_discoveries.extend(discoveries)
            
            # Log a sample of discoveries
            for i, discovery in enumerate(discoveries[:5]):
                logger.info(f"Discovery {i+1}: {discovery.get('summary', 'No summary')} (Type: {discovery.get('type', 'unknown')})")
                
            logger.info(f"Successfully investigated {repo['url']}")
            
        except Exception as e:
            logger.error(f"Error investigating {repo['url']}: {str(e)}")
    
    logger.info(f"Total discoveries across all repositories: {len(all_discoveries)}")
    return all_discoveries

if __name__ == "__main__":
    discoveries = test_github_investigation()
    
    # Write the discoveries to a file
    with open('github_discoveries.json', 'w') as f:
        import json
        json.dump(discoveries, f, indent=2)
        
    logger.info(f"Wrote {len(discoveries)} discoveries to github_discoveries.json")
    
    sys.exit(0)