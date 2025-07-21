"""
LLM Engine - OpenAI/Claude API handling and integration

This module contains functions for interacting with language models,
including handling API calls, retries, and context building.
"""

import os
import json
import time
import logging
import random
import re
import requests
from typing import List, Dict, Any, Optional, Tuple

# Configure logger
logger = logging.getLogger(__name__)

class LLMEngine:
    """Unified interface for different LLM providers with fallback support."""
    
    def __init__(self, use_claude: bool = True, use_openai: bool = False, use_gemini: bool = False):
        """
        Initialize LLM Engine with preferred models.
        
        Args:
            use_claude: Whether to use Claude
            use_openai: Whether to use OpenAI GPT
            use_gemini: Whether to use Google Gemini
        """
        self.use_claude = use_claude
        self.use_openai = use_openai
        self.use_gemini = use_gemini
        self.retry_count = 0
        self.max_retries = 3
        
        # Load API keys
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        
        # Validate we have at least one working API
        if not any([
            self.use_claude and self.anthropic_api_key,
            self.use_openai and self.openai_api_key,
            self.use_gemini and self.gemini_api_key
        ]):
            logger.warning("No valid LLM API configurations found")
            
    def run_prompt(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Run prompt with automatic fallback between models.
        
        Args:
            prompt: The prompt text
            max_tokens: Maximum tokens in response
            
        Returns:
            Model response as string
        """
        # Try primary model first
        if self.use_claude and self.anthropic_api_key:
            try:
                return self._call_claude(prompt)
            except Exception as e:
                logger.warning(f"Claude API error: {e}. Trying fallback.")
        
        # Try OpenAI fallback
        if self.use_openai and self.openai_api_key:
            try:
                return self._call_openai(prompt)
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}. Trying next fallback.")
        
        # Try Gemini fallback
        if self.use_gemini and self.gemini_api_key:
            try:
                return self._call_gemini(prompt)
            except Exception as e:
                logger.warning(f"Gemini API error: {e}. No more fallbacks available.")
                
        # If all attempts failed, return empty response
        logger.error("All LLM API calls failed")
        return ""
        
    def _call_claude(self, prompt: str) -> str:
        """Call the Claude API."""
        if not self.anthropic_api_key:
            logger.error("No Anthropic API key provided")
            return ""
            
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": "claude-2.0",
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant: ",
            "max_tokens_to_sample": 1500,
            "temperature": 0.5
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/complete",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            logger.error(f"Claude API error: {response.status_code}, {response.text}")
            raise Exception(f"Claude API returned status code {response.status_code}")
            
        result = response.json()
        return result.get("completion", "")
        
    def _call_openai(self, prompt: str) -> str:
        """Call the OpenAI API."""
        if not self.openai_api_key:
            logger.error("No OpenAI API key provided")
            return ""
            
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.5
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.status_code}, {response.text}")
            raise Exception(f"OpenAI API returned status code {response.status_code}")
            
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
    def _call_gemini(self, prompt: str) -> str:
        """Call the Google Gemini API."""
        if not self.gemini_api_key:
            logger.error("No Gemini API key provided")
            return ""
            
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 1500
            }
        }
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_api_key}",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            logger.error(f"Gemini API error: {response.status_code}, {response.text}")
            raise Exception(f"Gemini API returned status code {response.status_code}")
            
        result = response.json()
        
        # Extract the response text from the Gemini API response
        try:
            return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        except (KeyError, IndexError):
            logger.error("Failed to parse Gemini API response")
            return ""
            
    def analyze_text(self, text: str, objective: str) -> Dict[str, Any]:
        """
        Analyze text content for relevance to the objective.
        
        Args:
            text: Text content to analyze
            objective: Research objective
            
        Returns:
            Dictionary with analysis results
        """
        # Truncate very long content
        if len(text) > 8000:
            text = text[:8000] + "... [content truncated]"
            
        prompt = f"""
Analyze the following content for relevant information related to: "{objective}"

CONTENT:
{text}

Provide your analysis in JSON format with these fields:
- relevance_score: A number from 0-10 indicating how relevant this content is to the objective
- key_findings: A list of important discoveries in this content
- names: A list of any relevant person names mentioned
- entities: A list of relevant organizations, technologies, or projects mentioned
- dates: A list of any significant dates mentioned
- connections: A list of connections or relationships identified between entities
- research_leads: A list of promising research directions suggested by this content

JSON RESPONSE:
"""
        
        # Try to get analysis from LLM
        response = self.run_prompt(prompt)
        
        # Extract JSON
        try:
            json_str = self._extract_json(response)
            return json.loads(json_str)
        except:
            # If JSON extraction fails, return basic structure
            return {
                "relevance_score": 0,
                "key_findings": [],
                "names": [],
                "entities": [],
                "dates": [],
                "connections": [],
                "research_leads": []
            }
            
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text response."""
        # Try to extract JSON using regex pattern
        json_pattern = r'```json\n(.*?)\n```|```(.*?)```|\{.*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            if match.group(1):  # Matched the first pattern with json code block
                json_str = match.group(1)
            elif match.group(2):  # Matched the second pattern with generic code block
                json_str = match.group(2)
            else:  # Matched the third pattern with just braces
                json_str = match.group(0)
                
            # Try to repair incomplete JSON
            json_str = self._repair_json(json_str)
            return json_str
            
        # If no match found, return empty JSON
        return "{}"
        
    def _repair_json(self, json_text: str) -> str:
        """Attempt to repair malformed JSON."""
        # Remove any trailing commas before closing braces/brackets
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        # Ensure proper quoting of keys
        json_text = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_text)
        
        # Fix unquoted values (true, false, null are valid unquoted)
        json_text = re.sub(r':\s*([^"{}\[\],\s][^{}\[\],\s]*)\s*([,}])', r':"\1"\2', json_text)
        
        # Try to fix unbalanced braces/brackets
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        open_brackets = json_text.count('[')
        close_brackets = json_text.count(']')
        
        # Add missing closing braces
        if open_braces > close_braces:
            json_text += '}' * (open_braces - close_braces)
            
        # Add missing closing brackets
        if open_brackets > close_brackets:
            json_text += ']' * (open_brackets - close_brackets)
            
        return json_text

def get_llm_instance(use_claude=True, use_openai=False):
    """Get an instance of the LLM engine with specified preferences."""
    return LLMEngine(use_claude=use_claude, use_openai=use_openai)

def alternate_llm_instance(call_count):
    """Alternate between Claude and GPT for each call."""
    use_claude = (call_count % 2 == 0)
    return LLMEngine(use_claude=use_claude, use_openai=not use_claude)