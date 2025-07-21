#!/usr/bin/env python3
"""
Test script for Google Gemini integration.
"""

import logging
from llm_integration import LLMIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_gemini")

def test_gemini():
    """Test the Gemini LLM integration."""
    # Create an LLM integration instance with Gemini
    llm = LLMIntegration(use_claude=False, use_openai=False, use_gemini=True)
    
    # Test prompt
    prompt = "Explain in 3 paragraphs why blockchain technology is important for decentralized systems."
    
    # Call Gemini
    logger.info("Calling Gemini API...")
    response = llm._call_gemini(prompt)
    
    # Print the response
    logger.info(f"Gemini response:\n{response}")
    
    return response

def test_failover():
    """Test the failover from Claude to Gemini."""
    # Create an LLM integration instance with Claude primary and Gemini as backup
    llm = LLMIntegration(use_claude=True, use_openai=False, use_gemini=True)
    
    # Set an invalid Claude API key to force failover
    llm.claude_api_key = "invalid_key"
    
    # Test prompt
    prompt = "Explain in 3 paragraphs why blockchain technology is important for decentralized systems."
    
    # Call Claude (which should fail and fall back to Gemini)
    logger.info("Calling Claude API with invalid key (should fail over to Gemini)...")
    response = llm._call_claude(prompt)
    
    # Print the response
    logger.info(f"Fallback response:\n{response}")
    
    return response

if __name__ == "__main__":
    print("\n=== TESTING DIRECT GEMINI CALL ===\n")
    gemini_response = test_gemini()
    
    print("\n\n=== TESTING FAILOVER FROM CLAUDE TO GEMINI ===\n")
    failover_response = test_failover()
    
    print("\n=== TEST COMPLETE ===")