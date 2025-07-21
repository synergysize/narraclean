#!/usr/bin/env python3
"""
LLM Integration - Interface to language models for Narrahunt

This slim version imports core functionality from the modules directory.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import LLM functions from modules
from modules.llm_engine import LLMEngine, get_llm_instance, alternate_llm_instance

class LLMIntegration:
    """
    LLM Integration wrapper class for backward compatibility.
    
    This class maintains the original API while delegating to the
    modularized LLMEngine class.
    """
    
    def __init__(self, use_claude: bool = True, use_openai: bool = False, use_gemini: bool = False):
        """
        Initialize the LLM integration.
        
        Args:
            use_claude: Whether to use Claude
            use_openai: Whether to use OpenAI GPT
            use_gemini: Whether to use Google Gemini
        """
        self.engine = LLMEngine(use_claude=use_claude, use_openai=use_openai, use_gemini=use_gemini)
        
    def run_prompt(self, prompt: str, max_tokens: int = 1000) -> str:
        """Run a prompt through the LLM engine."""
        return self.engine.run_prompt(prompt, max_tokens)
        
    def analyze(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Analyze text content with the LLM."""
        objective = context or "Extract relevant information"
        return self.engine.analyze_text(text, objective)
        
    def generate_research_strategy(self, objective: str, entity: str) -> Dict[str, Any]:
        """
        Generate a research strategy for the given objective and entity.
        
        Args:
            objective: The research objective
            entity: The primary entity to investigate
            
        Returns:
            Dictionary with research strategy
        """
        prompt = f"""
You are a research strategist helping with an investigation on: "{objective}"

The primary entity of interest is: {entity}

Please provide a comprehensive research strategy, including:
1. Specific websites to investigate
2. Search queries to perform
3. GitHub repositories to check
4. Historical resources to examine
5. Key people to research

Format your response as a JSON object with these categories:
- website_targets: list of URLs to investigate
- search_targets: list of search queries
- github_targets: list of GitHub URLs
- wayback_targets: list of URLs to check in the Wayback Machine
- people_of_interest: list of names to research

JSON RESPONSE:
"""
        
        response = self.run_prompt(prompt)
        
        # Try to extract JSON
        try:
            json_str = self.engine._extract_json(response)
            return json.loads(json_str)
        except:
            # Return basic structure if JSON extraction fails
            return {
                "website_targets": [],
                "search_targets": [],
                "github_targets": [],
                "wayback_targets": [],
                "people_of_interest": []
            }

def main():
    """Main function to test LLM integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test LLM integration')
    parser.add_argument('--prompt', '-p', help='Prompt to run')
    parser.add_argument('--analyze', '-a', help='Text to analyze')
    parser.add_argument('--objective', '-o', default="Extract relevant information", help='Research objective')
    
    args = parser.parse_args()
    
    if not args.prompt and not args.analyze:
        parser.error('Either --prompt or --analyze must be provided')
        
    # Create LLM integration
    llm = LLMIntegration()
    
    if args.prompt:
        response = llm.run_prompt(args.prompt)
        print(f"LLM Response:\n{response}")
        
    if args.analyze:
        analysis = llm.analyze(args.analyze, args.objective)
        print(f"Analysis:\n{json.dumps(analysis, indent=2)}")
        
    return 0

if __name__ == '__main__':
    sys.exit(main())