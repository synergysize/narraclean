#!/usr/bin/env python3
"""
LLM Failover Implementation

This module demonstrates the failover mechanism to automatically switch from Claude to OpenAI
when Claude returns 401 errors or empty responses.
"""

import logging
from llm_integration import LLMIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_failover")

def call_llm_with_failover(prompt, use_claude_first=True):
    """
    Call an LLM with automatic failover from Claude to OpenAI if needed.
    
    Args:
        prompt: The prompt to send to the LLM
        use_claude_first: Whether to try Claude first (default: True)
        
    Returns:
        The LLM response text
    """
    if use_claude_first:
        # Try Claude first
        llm = LLMIntegration(use_claude=True, use_openai=False)
        result = llm._call_claude(prompt)
        
        # If Claude fails, try OpenAI as backup
        if not result or result.strip() == "{}":
            logger.warning("Claude failed, trying OpenAI as backup")
            backup_llm = LLMIntegration(use_claude=False, use_openai=True)
            result = backup_llm._call_openai(prompt)
    else:
        # Use OpenAI directly
        llm = LLMIntegration(use_claude=False, use_openai=True)
        result = llm._call_openai(prompt)
    
    return result

def main():
    """Test the LLM failover implementation."""
    # Example prompt
    prompt = "Generate a JSON object with fields: name, description, and rating (1-10)."
    
    # Test with Claude first, then fallback to OpenAI if needed
    response = call_llm_with_failover(prompt)
    
    logger.info(f"Response: {response}")
    
if __name__ == "__main__":
    main()