#!/usr/bin/env python3
"""
Test script for the modified Detective Agent logic.
"""

import logging
import json
from detective_agent import DetectiveAgent

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_research_strategy_generation():
    """Test the LLM strategy generation without fallback URLs."""
    print("Testing research strategy generation without fallbacks...")
    
    # Initialize the detective agent
    agent = DetectiveAgent(
        objective="Find name artifacts around Vitalik Buterin",
        entity="Vitalik Buterin",
        max_iterations=1
    )
    
    # Get the research strategy
    strategy = agent._get_initial_research_strategy()
    
    # Print the strategy
    print(f"\nStrategy generation results:")
    print(f"Sources: {len(strategy.get('sources', []))} URLs")
    print(f"Search queries: {len(strategy.get('search_queries', []))} queries")
    
    # Validate that the strategy is not empty
    if strategy.get('sources') or strategy.get('search_queries'):
        print("\n✅ Strategy generation successful - contained real sources/queries")
    else:
        print("\n❌ Strategy generation failed - no sources or queries found")
    
    # Verify no hardcoded fallbacks
    hardcoded_urls = [
        "https://github.com/vbuterin",
        "https://bitcointalk.org/index.php?action=profile;u=11772"
    ]
    
    uses_fallbacks = any(url in str(strategy.get('sources', [])) for url in hardcoded_urls)
    if not uses_fallbacks:
        print("✅ No hardcoded fallback URLs detected")
    else:
        print("❌ Fallback URLs still being used")
        
    return strategy

def test_empty_queue_handling():
    """Test handling of empty research queue."""
    print("\nTesting empty research queue handling...")
    
    # Create a mock agent with mock LLM that returns empty strategy
    class MockDetectiveAgent(DetectiveAgent):
        def _get_initial_research_strategy(self):
            print("Mocking empty strategy return")
            return {}
    
    # Initialize the mock agent
    agent = MockDetectiveAgent(
        objective="Find name artifacts around Vitalik Buterin",
        entity="Vitalik Buterin",
        max_iterations=1
    )
    
    # Start investigation
    print("Starting investigation with empty strategy...")
    discoveries = agent.start_investigation()
    
    # Should exit gracefully with empty list
    if discoveries == []:
        print("✅ Agent exited gracefully with empty research queue")
    else:
        print("❌ Agent did not handle empty queue properly")
    
    return discoveries

if __name__ == "__main__":
    test_research_strategy_generation()
    test_empty_queue_handling()