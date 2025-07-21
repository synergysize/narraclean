#!/usr/bin/env python3
"""
Test script for the modified Detective Agent.
"""

import logging
from detective_agent import DetectiveAgent

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_detective_agent():
    """Test the modified detective agent."""
    print("Testing detective agent with removed idle iterations limit and fallback URLs...")
    
    # Initialize the detective agent
    agent = DetectiveAgent(
        objective="Find name artifacts around Vitalik Buterin",
        entity="Vitalik Buterin",
        max_iterations=20,
        max_time_hours=0.1  # Set a short time for testing
    )
    
    # Start the investigation
    print("Starting investigation...")
    discoveries = agent.start_investigation()
    
    # Print results
    print(f"\nInvestigation complete. Found {len(discoveries)} discoveries.")
    
    # Print first few discoveries if any
    if discoveries:
        print("\nSample discoveries:")
        for i, discovery in enumerate(discoveries[:5]):
            print(f"{i+1}. {discovery.get('content', 'No content')} (type: {discovery.get('type', 'unknown')})")
    
    return discoveries

if __name__ == "__main__":
    test_detective_agent()