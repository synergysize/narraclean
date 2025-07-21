#!/usr/bin/env python3
"""
API Key Verification Utility for Narrahunt Phase 2.

This script checks if all required API keys are present and valid.
"""

import os
import sys
import json
import logging
import requests
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from config_loader import get_api_key, REQUIRED_API_KEYS, OPTIONAL_API_KEYS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('verify_api_keys')

def verify_claude_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Verify if the Claude API key is valid.
    
    Args:
        api_key: The Claude API key to verify
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not api_key:
        return False, "No API key provided"
    
    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "Hello, this is a test message to verify my API key."}
            ]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            return True, "Valid"
        elif response.status_code == 401:
            return False, f"Invalid API key (401 Unauthorized)"
        else:
            return False, f"Error: {response.status_code} - {response.text[:100]}"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def verify_openai_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Verify if the OpenAI API key is valid.
    
    Args:
        api_key: The OpenAI API key to verify
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not api_key:
        return False, "No API key provided"
    
    try:
        url = "https://api.openai.com/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return True, "Valid"
        elif response.status_code == 401:
            return False, f"Invalid API key (401 Unauthorized)"
        else:
            return False, f"Error: {response.status_code} - {response.text[:100]}"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def verify_all_api_keys() -> Dict[str, Dict[str, any]]:
    """
    Verify all API keys and return their status.
    
    Returns:
        Dictionary of API key statuses
    """
    # Load environment variables
    load_dotenv()
    
    results = {}
    
    # Verify required API keys
    for key_name in REQUIRED_API_KEYS:
        api_key = os.getenv(key_name)
        results[key_name] = {
            "present": bool(api_key),
            "valid": False,
            "message": "Not checked"
        }
        
        if api_key:
            # Verify the API key
            if key_name == 'CLAUDE_API_KEY':
                valid, message = verify_claude_api_key(api_key)
                results[key_name]["valid"] = valid
                results[key_name]["message"] = message
            elif key_name == 'OPENAI_API_KEY':
                valid, message = verify_openai_api_key(api_key)
                results[key_name]["valid"] = valid
                results[key_name]["message"] = message
    
    # Check optional API keys
    for key_name in OPTIONAL_API_KEYS:
        api_key = os.getenv(key_name)
        results[key_name] = {
            "present": bool(api_key),
            "required": False,
            "message": "Optional key"
        }
    
    return results

if __name__ == "__main__":
    logger.info("Verifying API keys...")
    results = verify_all_api_keys()
    
    # Print results in a table format
    logger.info("\nAPI Key Verification Results:")
    logger.info("-" * 80)
    logger.info(f"{'Key Name':<25} {'Present':<10} {'Valid':<10} {'Message':<35}")
    logger.info("-" * 80)
    
    # Required keys first
    for key_name in REQUIRED_API_KEYS:
        result = results[key_name]
        logger.info(f"{key_name:<25} {'✓' if result['present'] else '✗':<10} {'✓' if result.get('valid', False) else '✗':<10} {result['message']:<35}")
    
    logger.info("-" * 80)
    
    # Optional keys
    for key_name in OPTIONAL_API_KEYS:
        result = results[key_name]
        logger.info(f"{key_name:<25} {'✓' if result['present'] else '-':<10} {'N/A':<10} {result['message']:<35}")
    
    logger.info("-" * 80)
    
    # Check if all required keys are valid
    all_valid = all(results[key_name].get('valid', False) for key_name in REQUIRED_API_KEYS)
    
    if all_valid:
        logger.info("\n✅ All required API keys are present and valid.")
        sys.exit(0)
    else:
        logger.error("\n❌ Some required API keys are missing or invalid.")
        sys.exit(1)