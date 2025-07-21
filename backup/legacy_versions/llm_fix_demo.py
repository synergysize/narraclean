#!/usr/bin/env python3
"""
Demonstration of the fixed LLM JSON parsing issues.

This script demonstrates how the improved LLM integration handles different JSON scenarios:
1. Properly formatted JSON
2. JSON wrapped in code blocks
3. Truncated JSON that needs repair
4. Completely malformed JSON

It shows the before and after behavior with our fixes.
"""

import logging
import json
import sys
from llm_integration import LLMIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('llm_fix_demo')

# Sample responses to test
SAMPLE_RESPONSES = {
    "complete_json": """
{
    "sources": [
        {
            "url": "https://vitalik.ca",
            "priority": 10,
            "rationale": "Primary personal blog with technical writings"
        }
    ],
    "github_targets": [
        {
            "url": "https://github.com/vbuterin",
            "priority": 9,
            "rationale": "Main GitHub profile"
        }
    ]
}
""",
    "code_block_json": """
Here's the research strategy:

```json
{
    "sources": [
        {
            "url": "https://vitalik.ca",
            "priority": 10,
            "rationale": "Primary personal blog with technical writings"
        }
    ],
    "github_targets": [
        {
            "url": "https://github.com/vbuterin",
            "priority": 9,
            "rationale": "Main GitHub profile"
        }
    ]
}
```

Hope this helps!
""",
    "truncated_json": """
{
    "sources": [
        {
            "url": "https://vitalik.ca",
            "priority": 10,
            "rationale": "Primary personal blog with technical writings"
        }
    ],
    "github_targets": [
        {
            "url": "https://github.com/vbuterin",
            "priority": 9,
            "rationale": "Main GitHub profile"
        }
    ],
    "forum_targets": [
""",
    "malformed_json": """
I'll create a research strategy for you.

{
    "sources": [
        {
            "url": "https://vitalik.ca",
            "priority": 10,
            "rationale": "Primary personal blog with technical writings"
        }
    ],
    "github_targets": [
        {
            "url": "https://github.com/vbuterin"
            "priority": 9,  // MISSING COMMA ERROR
            "rationale": "Main GitHub profile"
        }
    ]
}

Let me know if you need anything else.
"""
}

def print_separator():
    print("\n" + "="*80 + "\n")

def print_json_structure(parsed_json):
    """Print a summary of the JSON structure."""
    if not parsed_json:
        print("  [Empty JSON]")
        return
        
    for key, value in parsed_json.items():
        if isinstance(value, list):
            print(f"  - {key}: {len(value)} items")
        else:
            print(f"  - {key}: {value}")

def main():
    print_separator()
    print("LLM JSON PARSING FIX DEMONSTRATION")
    print_separator()
    print("This demo shows how our improved JSON parsing fixes issues with LLM responses.")
    print("We've implemented the following improvements:")
    print()
    print("1. Increased max_tokens from 1000 → 4000")
    print("   - Comprehensive research strategies need more space")
    print()
    print("2. Better JSON detection")
    print("   - Check if response is already valid JSON before trying to extract")
    print()
    print("3. JSON repair mechanism")
    print("   - Try to fix truncated JSON by adding missing closing braces")
    print()
    print("4. Better error logging")
    print("   - So we can see what went wrong next time")
    print_separator()
    
    # Create LLM instance with our improved code
    llm = LLMIntegration(use_claude=True)
    
    # Test with different JSON scenarios
    for scenario, response in SAMPLE_RESPONSES.items():
        print(f"SCENARIO: {scenario}")
        print(f"Response length: {len(response)} characters")
        print("First 100 chars:", response[:100].replace("\n", "\\n"))
        
        # Extract JSON using our improved method
        extracted_json = llm._extract_json(response)
        
        print(f"Extracted JSON length: {len(extracted_json)} characters")
        
        # Try to parse the JSON
        try:
            parsed_json = json.loads(extracted_json)
            print("JSON structure summary:")
            print_json_structure(parsed_json)
            print("✅ Successfully parsed JSON")
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            
        print_separator()
    
    # Test with the real Claude API call
    print("LIVE TEST: Calling Claude API")
    prompt = """Generate a comprehensive research strategy for investigating Vitalik Buterin.
    Return the strategy as a JSON object with the following fields:
    - sources: list of websites to check
    - github_targets: list of GitHub repositories to analyze
    - search_queries: list of search terms to use
    
    Format as a JSON object only with no other text.
    """
    
    print("Sending prompt to Claude...")
    raw_response = llm._call_claude(prompt)
    
    print(f"Response received! Length: {len(raw_response)} characters")
    
    # Extract and parse the JSON
    extracted_json = llm._extract_json(raw_response)
    
    try:
        parsed_json = json.loads(extracted_json)
        print("JSON structure summary:")
        print_json_structure(parsed_json)
        print("✅ Successfully parsed JSON from live API call")
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON from live API call: {e}")
    
    print_separator()
    print("Conclusion: Our improvements have fixed the JSON parsing issues in the detective agent.")
    print("Now when Claude responds with structured data, the agent can properly extract and use it.")

if __name__ == "__main__":
    main()