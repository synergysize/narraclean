#!/usr/bin/env python3
"""
Test error handling for LLM and Wayback Machine API integration.
"""

import json
import logging
import sys
from llm_integration import LLMIntegration
from wayback_integration import WaybackMachine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_error_handling')

def test_llm_error_handling():
    """Test LLM integration error handling."""
    logger.info("Testing LLM integration error handling...")
    
    # Initialize LLM integration
    llm = LLMIntegration(use_claude=True, use_openai=True)
    
    # Test with empty response from Claude
    logger.info("Testing Claude API empty response handling...")
    result = llm._call_claude("Test prompt")
    logger.info(f"Claude result: {result}")
    
    # Test with empty response from OpenAI
    logger.info("Testing OpenAI API empty response handling...")
    result = llm._call_openai("Test prompt")
    logger.info(f"OpenAI result: {result}")
    
    logger.info("LLM error handling tests completed")

def test_wayback_error_handling():
    """Test Wayback Machine integration error handling."""
    logger.info("Testing Wayback Machine integration error handling...")
    
    # Initialize Wayback Machine integration
    wayback = WaybackMachine()
    
    # Test with invalid URL
    logger.info("Testing Wayback Machine with invalid URL...")
    result = wayback.check_availability("thisisaninvalidurl.example")
    logger.info(f"Wayback result: {json.dumps(result, indent=2)}")
    
    logger.info("Wayback error handling tests completed")

def main():
    """Main function to run error handling tests."""
    logger.info("Starting error handling tests...")
    
    test_llm_error_handling()
    test_wayback_error_handling()
    
    logger.info("All error handling tests completed")

if __name__ == "__main__":
    main()