#!/usr/bin/env python3
"""
LLM Integration Module for Narrahunt Phase 2.

Provides integration with LLM services (Anthropic Claude and OpenAI) for 
analyzing and enhancing discoveries.
"""

import os
import json
import logging
import requests
import time
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
from config.config_loader import get_api_key

# Configure logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_dir, 'logs', 'llm_integration.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('llm_integration')

# Load environment variables with explicit path
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)
logger.info(f"Loading .env from: {env_path}")
logger.info(f".env exists: {os.path.exists(env_path)}")

# LLM API Keys loaded from environment
CLAUDE_API_KEY = get_api_key('CLAUDE_API_KEY') or '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'
OPENAI_API_KEY = get_api_key('OPENAI_API_KEY') or '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'
GOOGLE_API_KEY = get_api_key('GOOGLE_API_KEY') or 'AIzaSyAa__kXU5Nr63Kzgzfl8hXpRG2rAX8JGHM'

class LLMIntegration:
    """
    Provides integration with LLM services for content analysis and enhancement.
    """
    
    def __init__(self, use_claude: bool = True, use_openai: bool = False, use_gemini: bool = False):
        """
        Initialize the LLM integration module.
        
        Args:
            use_claude: Whether to use Claude as the LLM
            use_openai: Whether to use OpenAI as the LLM
            use_gemini: Whether to use Google Gemini as the LLM
            
        Raises:
            ValueError: If the selected LLM service API key is not available
        """
        self.use_claude = use_claude
        self.use_openai = use_openai
        self.use_gemini = use_gemini
        
        if not use_claude and not use_openai and not use_gemini:
            logger.warning("No LLM service selected. Defaulting to Claude.")
            self.use_claude = True
        
        # Initialize API keys
        self.claude_api_key = CLAUDE_API_KEY
        self.openai_api_key = OPENAI_API_KEY
        self.google_api_key = GOOGLE_API_KEY
        
        # Validate that we have the necessary API keys
        if self.use_claude and not self.claude_api_key:
            raise ValueError("Claude API key not found. Please add CLAUDE_API_KEY to your .env file.")
        
        if self.use_openai and not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
            
        if self.use_gemini and not self.google_api_key:
            raise ValueError("Google API key not found. Please add GOOGLE_API_KEY to your .env file.")
        
        # Cache for API responses
        self.response_cache = {}
        
        logger.info(f"LLM Integration initialized. Using Claude: {use_claude}, Using OpenAI: {use_openai}, Using Gemini: {use_gemini}")
    
    def analyze(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze text content using the selected LLM.
        
        Args:
            text: The text to analyze
            context: Optional context for the analysis
            
        Returns:
            Dictionary with analysis results
        """
        if not text:
            return {
                "entities": [],
                "sentiment": "neutral",
                "relevance_score": 0.0,
                "narrative_score": 0.0,
                "summary": ""
            }
        
        # Prepare prompt
        if context:
            prompt = f"Context: {context}\n\nText to analyze: {text}\n\n"
        else:
            prompt = f"Text to analyze: {text}\n\n"
        
        prompt += """Please analyze this text and return the following:
1. Entities: List any people, organizations, projects, or other named entities
2. Sentiment: Overall sentiment (positive, negative, neutral)
3. Relevance Score: How relevant is this text to the context (0.0 to 1.0)
4. Narrative Value: How valuable is this for a narrative (0.0 to 1.0)
5. Summary: A brief 1-2 sentence summary

Format your response as JSON with these fields.
"""
        
        # Calculate cache key
        cache_key = f"{hash(prompt)}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Get response from LLM
        if self.use_claude:
            result = self._call_claude(prompt)
            # If Claude fails, try backup options
            if not result or result.strip() == "{}":
                logger.warning("Claude failed, trying backup LLM")
                if self.use_openai:
                    result = self._call_openai(prompt)
                elif self.use_gemini:
                    result = self._call_gemini(prompt)
        elif self.use_openai:
            result = self._call_openai(prompt)
            # If OpenAI fails, try Gemini as backup
            if not result or result.strip() == "{}":
                logger.warning("OpenAI failed, trying Gemini as backup")
                if self.use_gemini:
                    result = self._call_gemini(prompt)
        elif self.use_gemini:
            result = self._call_gemini(prompt)
        else:
            logger.error("No LLM service available")
            return {
                "entities": [],
                "sentiment": "neutral",
                "relevance_score": 0.5,
                "narrative_score": 0.5,
                "summary": "No LLM service available for analysis."
            }
        
        # Parse the response
        try:
            # Try to extract JSON from the response
            json_str = self._extract_json(result)
            data = json.loads(json_str)
            
            # Ensure all fields are present
            analysis = {
                "entities": data.get("entities", []),
                "sentiment": data.get("sentiment", "neutral"),
                "relevance_score": float(data.get("relevance_score", 0.5)),
                "narrative_score": float(data.get("narrative_score", 0.5)),
                "summary": data.get("summary", "")
            }
            
            # Cache the result
            self.response_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Raw response: {result}")
            
            # Return a fallback analysis
            return {
                "entities": [],
                "sentiment": "neutral",
                "relevance_score": 0.5,
                "narrative_score": 0.5,
                "summary": "Failed to analyze text."
            }
    
    def _call_claude(self, prompt: str) -> str:
        """
        Call the Claude API.
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            Claude's response text
        """
        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.claude_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            # Increase max_tokens to handle comprehensive research strategies
            data = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 4000,  # Increased from 1000 to handle larger research strategies
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            logger.debug(f"Calling Claude API with prompt length: {len(prompt)} characters")
            response = requests.post(url, headers=headers, json=data, timeout=60)  # Increased timeout
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'content' in result and len(result['content']) > 0:
                        content = result['content'][0]['text']
                        logger.info(f"Successfully received response from Claude (length: {len(content)} characters)")
                        
                        # Check if the response is incomplete JSON
                        if content.strip().startswith('{') and not content.strip().endswith('}'):
                            logger.warning("Claude returned incomplete JSON. Attempting to repair.")
                            # Try to repair by closing any unclosed structures
                            content = self._repair_incomplete_json(content)
                        
                        return content
                    else:
                        logger.error("Invalid Claude API response structure")
                        return "{}"
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.error(f"Error parsing Claude response: {e}")
                    return "{}"
            elif response.status_code == 401:
                logger.warning(f"Claude authentication failed (401) - automatically falling back to alternative LLM")
                # If we have OpenAI access, use it as fallback
                if self.openai_api_key:
                    logger.info("Falling back to OpenAI")
                    return self._call_openai(prompt)
                # If we have Gemini access, use it as secondary fallback
                elif self.google_api_key:
                    logger.info("Falling back to Gemini")
                    return self._call_gemini(prompt)
                else:
                    logger.error("No alternative LLM API keys available for fallback")
                    return "{}"
            else:
                logger.error(f"Claude API error: {response.status_code} - {response.text}")
                return "{}"
                
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return "{}"
            
    def _repair_incomplete_json(self, json_text: str) -> str:
        """
        Attempt to repair incomplete JSON by closing any open structures.
        
        Args:
            json_text: Incomplete JSON text
            
        Returns:
            Repaired JSON text (best effort)
        """
        # Count open/close brackets and braces
        open_curly = json_text.count('{')
        close_curly = json_text.count('}')
        open_square = json_text.count('[')
        close_square = json_text.count(']')
        
        logger.debug(f"JSON repair: Found {open_curly} open curly braces, {close_curly} close curly braces")
        logger.debug(f"JSON repair: Found {open_square} open square brackets, {close_square} close square brackets")
        
        # Add missing closing braces/brackets
        repaired = json_text
        
        # If we're in the middle of an incomplete object value, add a closing quote
        last_colon = repaired.rfind(':')
        last_comma = repaired.rfind(',')
        last_quote = repaired.rfind('"')
        
        if last_colon > last_comma and last_colon > last_quote:
            # We have a dangling property without a value
            repaired += ' ""'
            
        # Close any unclosed arrays
        for _ in range(open_square - close_square):
            if repaired.rstrip().endswith(','):
                # Remove trailing comma if present
                repaired = repaired.rstrip()[:-1]
            repaired += ']'
            
        # Close any unclosed objects
        for _ in range(open_curly - close_curly):
            if repaired.rstrip().endswith(','):
                # Remove trailing comma if present
                repaired = repaired.rstrip()[:-1]
            repaired += '}'
            
        logger.debug(f"JSON repair: Original length {len(json_text)}, Repaired length {len(repaired)}")
        
        # Verify the repaired JSON is valid
        try:
            json.loads(repaired)
            logger.info("Successfully repaired incomplete JSON")
            return repaired
        except Exception as e:
            logger.error(f"Failed to repair JSON: {e}")
            return json_text  # Return original if repair failed
    
    def _call_gemini(self, prompt: str) -> str:
        """
        Call the Google Gemini API.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            Gemini's response text
        """
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            headers = {"Content-Type": "application/json"}
            
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 1000}
            }
            
            response = requests.post(f"{url}?key={self.google_api_key}", 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return "{}"
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return "{}"
            
    def _call_openai(self, prompt: str) -> str:
        """
        Call the OpenAI API.
        
        Args:
            prompt: The prompt to send to OpenAI
            
        Returns:
            OpenAI's response text
        """
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4-turbo",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content']
                        logger.info("Successfully received response from OpenAI")
                        return content
                    else:
                        logger.error("Invalid OpenAI API response structure")
                        return "{}"
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.error(f"Error parsing OpenAI response: {e}")
                    return "{}"
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return "{}"
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "{}"
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from a text response.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            The extracted JSON string
        """
        # Log the raw text for debugging
        logger.debug(f"Extracting JSON from text (length: {len(text)})")
        
        # Check if the raw text is already valid JSON
        try:
            json.loads(text)
            logger.debug("Text is already valid JSON")
            return text.strip()
        except:
            logger.debug("Text is not valid JSON on its own, attempting extraction")
        
        # Look for JSON block with code fence
        if "```json" in text and "```" in text.split("```json", 1)[1]:
            logger.debug("Extracting JSON from ```json code block")
            json_text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            try:
                json.loads(json_text)  # Test if it's valid JSON
                logger.info("Successfully extracted JSON from ```json code block")
                return json_text
            except json.JSONDecodeError as e:
                logger.warning(f"JSON in ```json code block is invalid: {e}")
        elif "```" in text and "```" in text.split("```", 1)[1]:
            logger.debug("Extracting JSON from ``` code block")
            json_text = text.split("```", 1)[1].split("```", 1)[0].strip()
            try:
                json.loads(json_text)  # Test if it's valid JSON
                logger.info("Successfully extracted JSON from ``` code block")
                return json_text
            except json.JSONDecodeError as e:
                logger.warning(f"Code block content is not valid JSON: {e}")
        
        # If no JSON block, look for the first { and last }
        if "{" in text and "}" in text:
            logger.debug("Extracting JSON from curly braces")
            start = text.find("{")
            end = text.rfind("}") + 1
            json_text = text[start:end].strip()
            try:
                json.loads(json_text)  # Test if it's valid JSON
                logger.info("Successfully extracted JSON from curly braces")
                return json_text
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from curly braces: {e}")
                
                # Try a more aggressive approach to fix common issues
                try:
                    # Sometimes the JSON is cut off - try to find the last complete object
                    last_object_end = json_text.rfind("}")
                    if last_object_end > 0:
                        # Find the matching opening brace
                        open_braces = 0
                        for i in range(last_object_end, -1, -1):
                            if json_text[i] == "}":
                                open_braces += 1
                            elif json_text[i] == "{":
                                open_braces -= 1
                                if open_braces == 0:
                                    json_text = json_text[i:last_object_end+1]
                                    json.loads(json_text)  # Test if it's valid JSON
                                    logger.debug("Successfully extracted JSON using brace matching")
                                    return json_text
                except:
                    logger.debug("Failed to extract JSON using brace matching")
        
        # If all else fails, return an empty JSON object
        logger.error("Could not extract valid JSON, returning empty object")
        # Log a sample of the text to help with debugging
        if len(text) > 200:
            logger.debug(f"Text sample (first 100 chars): {text[:100]}")
            logger.debug(f"Text sample (last 100 chars): {text[-100:]}")
        else:
            logger.debug(f"Full text: {text}")
        return '{}'
    
    def generate_research_strategy(self, objective: str, entity: str) -> Dict[str, List[str]]:
        """
        Generate a research strategy for a given objective and entity.
        
        Args:
            objective: The research objective
            entity: The target entity
            
        Returns:
            Dictionary with research strategy
        """
        prompt = f"""
Generate a detailed research strategy for the following objective:

Objective: {objective}
Entity: {entity}

Please identify:
1. Key sources to check (websites, forums, archives)
2. Specific search queries to use
3. Types of information to look for
4. Historical periods or events to focus on

Format your response as a JSON object with these sections.
"""
        
        if self.use_claude:
            result = self._call_claude(prompt)
        elif self.use_openai:
            result = self._call_openai(prompt)
        else:
            logger.error("No LLM service available")
            return {
                "sources": [],
                "search_queries": [],
                "information_types": [],
                "time_periods": []
            }
        
        try:
            json_str = self._extract_json(result)
            data = json.loads(json_str)
            
            strategy = {
                "sources": data.get("sources", []),
                "search_queries": data.get("search_queries", []),
                "information_types": data.get("information_types", []),
                "time_periods": data.get("time_periods", [])
            }
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error parsing research strategy: {e}")
            return {
                "sources": [],
                "search_queries": [],
                "information_types": [],
                "time_periods": []
            }
    
    def evaluate_discovery(self, discovery: Dict[str, Any], objective: str) -> float:
        """
        Evaluate a discovery for its narrative value.
        
        Args:
            discovery: The discovery to evaluate
            objective: The research objective
            
        Returns:
            Narrative value score (0.0 to 1.0)
        """
        prompt = f"""
Evaluate this discovery for its narrative value:

Objective: {objective}
Content: {discovery.get('content', '')}
Source: {discovery.get('url', 'Unknown')}

On a scale of 0.0 to 1.0, how valuable is this discovery for building a narrative?
Consider factors like:
- Uniqueness
- Historical significance
- Emotional impact
- Connection to the objective
- Potential for storytelling

Return only a number between 0.0 and 1.0.
"""
        
        if self.use_claude:
            result = self._call_claude(prompt)
        elif self.use_openai:
            result = self._call_openai(prompt)
        else:
            logger.error("No LLM service available")
            return 0.5
        
        try:
            # Extract the number from the response
            import re
            number_match = re.search(r'(\d+\.\d+|\d+)', result)
            if number_match:
                score = float(number_match.group(1))
                # Ensure score is between 0.0 and 1.0
                score = max(0.0, min(1.0, score))
                return score
            else:
                return 0.5
        except Exception as e:
            logger.error(f"Error parsing evaluation score: {e}")
            return 0.5

# Test the LLM integration
if __name__ == "__main__":
    print("Testing LLM Integration")
    
    llm = LLMIntegration(use_claude=True, use_openai=False)
    
    # Test text analysis
    test_text = """
    Vitalik Buterin created Ethereum in 2015 after being involved in the Bitcoin community.
    He published the Ethereum whitepaper in 2013 and launched the network with co-founders
    including Gavin Wood and Joseph Lubin. The project has since become one of the most
    important blockchain platforms for smart contracts and decentralized applications.
    """
    
    analysis = llm.analyze(test_text, context="Looking for information about Ethereum founders")
    print("\nText Analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Test research strategy generation
    strategy = llm.generate_research_strategy(
        "Find name artifacts around Vitalik Buterin",
        "Vitalik Buterin"
    )
    print("\nResearch Strategy:")
    print(json.dumps(strategy, indent=2))
    
    # Test discovery evaluation
    discovery = {
        "content": "Vitalik Buterin used the username 'vitalik_btc' on the Bitcoin forum before creating Ethereum.",
        "url": "https://bitcointalk.org/index.php?action=profile;u=11772"
    }
    
    score = llm.evaluate_discovery(discovery, "Find name artifacts around Vitalik Buterin")
    print(f"\nDiscovery Evaluation Score: {score}")
    
    print("\nLLM Integration test complete.")