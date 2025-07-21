#!/usr/bin/env python3
"""
Test script to debug the LLM integration's JSON parsing.
"""

import os
import json
import logging
from llm_integration import LLMIntegration

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('test_llm')

def test_llm_json_extraction():
    """Test the LLM's JSON extraction logic with the comprehensive research strategy prompt."""
    
    # Create LLM instance (using Claude)
    llm = LLMIntegration(use_claude=True, use_openai=False)
    
    # Sample entity to investigate
    entity = "Vitalik Buterin"
    objective = "Find name artifacts around Vitalik Buterin"
    
    # Recreate the prompt from detective_agent.py
    prompt = f"""
You are a master research strategist planning a focused, methodical investigation of: "{objective}"

MISSION: Create a comprehensive TODO list of ALL potential high-value targets for finding "narrative artifacts" - historical names, code, wallets, or terminology that can create compelling backstories for cryptocurrency token launches. We need a methodical, depth-first exploration strategy.

SUCCESSFUL NARRATIVE EXAMPLES:
- SimpleToken: Contract code found in 2016 Ethereum.org archives → launched using that exact code
- Simple Wallet: Private key discovered in same archives → token launched from that wallet  
- Buckazood: 1990s Sierra game fictional currency with Bitcoin-like logo → speculation about Hal Finney
- Signa: Old Ethereum logo (two overlapping Sigma symbols) → launched as historical tribute
- Zeus: Matt Furie's dog name found by digging through his website photos → personal connection
- PEPE1.0: Original Pepe drawing found in Matt Furie's court evidence vs Alex Jones → legal archaeology  
- Strawberry: Female Pepe discovered in Matt Furie's unpublished book → insider knowledge
- Casinu: Token name later referenced by Vitalik in academic paper → retroactive validation
- NexF: FBI token for busting market makers, revealed later as evidence → government conspiracy

MY CAPABILITIES:
✓ Crawl current websites and extract content
✓ Wayback Machine access (2013-2025, especially 2013-2017 crypto goldmine)
✓ GitHub repository mining and commit history analysis  
✓ Targeted search queries across the web
✓ Forum profile investigation and historical posts
✓ Court documents and academic paper analysis
✓ Archive snapshot analysis from key time periods
✓ Social media archaeology and deleted content recovery

TARGET: {entity}

WHAT I NEED:
A COMPREHENSIVE RESEARCH PLAN with ALL potential targets prioritized (not just a few examples):

1. Complete list of ALL potential GitHub repositories (not just main profile)
2. ALL relevant forum profiles and specific threads to investigate
3. ALL personal websites, blogs, and social media profiles
4. ALL significant historical snapshots to check in Wayback Machine
5. ALL targeted search queries that could reveal unique discoveries
6. ALL known projects, collaborations, and early involvement areas

For EACH target, provide:
- Priority rating (1-10) based on likelihood of containing valuable artifacts
- Rationale for investigation (WHY this target may contain valuable information)
- What specific artifacts we might find there
- Time periods of highest interest

EXCLUDE: Obvious biographical info, well-known facts, their actual name variations

FORMAT: Return a comprehensive JSON research plan with sources, github_targets, wayback_targets, search_queries, and forum_targets arrays. Each item should have a priority rating and investigation rationale.

Please respond with ONLY a JSON object in this exact format:
{{
    "sources": [
        {{
            "url": "https://vitalik.ca",
            "priority": 10,
            "rationale": "Primary personal blog with technical writings dating back to 2013",
            "potential_artifacts": ["early project names", "wallet addresses", "unpublished ideas"]
        }}
    ],
    "github_targets": [
        {{
            "url": "https://github.com/vbuterin/ethereum",
            "priority": 9,
            "rationale": "Original Ethereum repository with early commits and code",
            "potential_artifacts": ["test addresses", "contract snippets", "prototype names"]
        }}
    ],
    "wayback_targets": [
        {{
            "url": "https://ethereum.org",
            "priority": 8,
            "years": [2014, 2015, 2016],
            "rationale": "Early Ethereum website versions with potential keys, test addresses",
            "potential_artifacts": ["test wallet keys", "original branding elements", "early team members"]
        }}
    ],
    "search_queries": [
        {{
            "query": "Vitalik Buterin early projects before Ethereum",
            "priority": 7,
            "rationale": "Find pre-Ethereum projects that might contain valuable artifacts",
            "potential_artifacts": ["project names", "usernames", "collaboration details"]
        }}
    ],
    "forum_targets": [
        {{
            "url": "https://bitcointalk.org/index.php?action=profile;u=11772",
            "priority": 10,
            "rationale": "Vitalik's Bitcoin forum profile with early crypto discussions",
            "potential_artifacts": ["early opinions", "forgotten predictions", "username variations"]
        }}
    ],
    "key_time_periods": [
        {{
            "period": "2011-2013",
            "description": "Bitcoin Magazine and early crypto involvement",
            "priority": 9
        }},
        {{
            "period": "2013-2015",
            "description": "Ethereum conception and launch",
            "priority": 10
        }}
    ]
}}

Return ONLY the JSON object, no other text. Make this a COMPREHENSIVE list of ALL potential targets, not just a few examples.
"""
    
    # Call Claude API
    logger.info("Calling Claude API...")
    raw_response = llm._call_claude(prompt)
    
    # Log the raw response
    logger.info("Raw Claude response received. Length: %d characters", len(raw_response))
    logger.debug("Raw response: %s", raw_response)
    
    # Attempt to extract JSON
    logger.info("Extracting JSON...")
    extracted_json = llm._extract_json(raw_response)
    logger.info("Extracted JSON length: %d characters", len(extracted_json))
    
    # Try to parse the JSON
    try:
        data = json.loads(extracted_json)
        logger.info("Successfully parsed JSON")
        
        # Log structure summary
        logger.info("JSON structure summary:")
        logger.info("- sources: %d items", len(data.get("sources", [])))
        logger.info("- github_targets: %d items", len(data.get("github_targets", [])))
        logger.info("- wayback_targets: %d items", len(data.get("wayback_targets", [])))
        logger.info("- search_queries: %d items", len(data.get("search_queries", [])))
        logger.info("- forum_targets: %d items", len(data.get("forum_targets", [])))
        logger.info("- key_time_periods: %d items", len(data.get("key_time_periods", [])))
        
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON: %s", e)
        logger.debug("Extracted JSON content: %s", extracted_json)
    
    # Save raw response and extracted JSON to files for inspection
    with open("raw_claude_response.txt", "w") as f:
        f.write(raw_response)
    
    with open("extracted_json.txt", "w") as f:
        f.write(extracted_json)
    
    logger.info("Test complete. Raw response and extracted JSON saved to files.")

if __name__ == "__main__":
    test_llm_json_extraction()