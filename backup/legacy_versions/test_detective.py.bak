#!/usr/bin/env python3
"""
Test script for the Detective Agent JSON parsing fix.
"""

import json
import logging
from detective_agent import DetectiveAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_research_strategy():
    """Test the initial research strategy generation."""
    print("Testing initial research strategy generation...")
    
    # Initialize the detective agent
    agent = DetectiveAgent(
        objective="Find name artifacts around Vitalik Buterin",
        entity="Vitalik Buterin",
        max_iterations=10
    )
    
    # Get the initial research strategy
    strategy = agent._get_initial_research_strategy()
    
    # Print the strategy in a readable format
    print("\nResearch Strategy:")
    print(json.dumps(strategy, indent=2))
    
    # Check if the strategy contains the expected fields
    if strategy.get('sources') and len(strategy.get('sources', [])) > 0:
        print("\n✅ Sources found successfully!")
    else:
        print("\n❌ No sources found in the strategy.")
    
    if strategy.get('search_queries') and len(strategy.get('search_queries', [])) > 0:
        print("✅ Search queries found successfully!")
    else:
        print("❌ No search queries found in the strategy.")
    
    if strategy.get('information_types') and len(strategy.get('information_types', [])) > 0:
        print("✅ Information types found successfully!")
    else:
        print("❌ No information types found in the strategy.")
    
    if strategy.get('time_periods') and len(strategy.get('time_periods', [])) > 0:
        print("✅ Time periods found successfully!")
    else:
        print("❌ No time periods found in the strategy.")
    
    return strategy

if __name__ == "__main__":
    test_research_strategy()