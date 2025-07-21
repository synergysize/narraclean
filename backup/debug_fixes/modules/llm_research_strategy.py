#!/usr/bin/env python3
"""
LLM Research Strategy Generator

This module uses an LLM to generate research strategies based on narrative objectives.
It converts high-level objectives into specific, crawlable URLs and search queries.
"""

import os
import re
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
from datetime import datetime
from dotenv import load_dotenv
from config.config_loader import get_api_key

# Configure logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_dir, 'logs', 'llm_strategy.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('llm_research_strategy')

class LLMResearchStrategy:
    """
    Uses LLM to generate research strategies for narrative objectives.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM Research Strategy generator.
        
        Args:
            api_key: Optional API key for the LLM service
            
        Raises:
            ValueError: If no API key is provided or found in environment
        """
        # Load environment variables
        load_dotenv()
        
        # Try to get the API key in order of precedence:
        # 1. Explicitly provided api_key parameter
        # 2. CLAUDE_API_KEY from environment (via config_loader)
        # 3. LLM_API_KEY from environment (legacy)
        self.api_key = api_key or get_api_key('CLAUDE_API_KEY') or os.environ.get('LLM_API_KEY')
        
        if not self.api_key:
            raise ValueError("No LLM API key provided. Please set CLAUDE_API_KEY in your .env file.")
        
        # Load strategy templates
        self.strategy_templates_path = os.path.join(base_dir, 'config', 'strategy_templates.json')
        self.strategy_templates = self._load_strategy_templates()
        
        # Track generated strategies for caching
        self.strategy_cache = {}
    
    def _load_strategy_templates(self) -> Dict[str, Any]:
        """Load strategy templates from file or create default ones."""
        if os.path.exists(self.strategy_templates_path):
            try:
                with open(self.strategy_templates_path, 'r') as f:
                    return json.load(f)
            except:
                logger.error(f"Error loading strategy templates from {self.strategy_templates_path}")
        
        # No hardcoded templates - use only LLM strategy
        default_templates = {}
        
        # Save default templates
        os.makedirs(os.path.dirname(self.strategy_templates_path), exist_ok=True)
        with open(self.strategy_templates_path, 'w') as f:
            json.dump(default_templates, f, indent=2)
        
        return default_templates
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API to generate a research strategy.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response text
        """
        # Try to import the LLM integration module
        try:
            from llm_integration import LLMIntegration
            llm = LLMIntegration(use_claude=True, use_openai=False)
            logger.info("Using real LLM integration with Claude API")
            
            # Using Claude to analyze the prompt directly
            # Extract the entity from the prompt
            import re
            entity_match = re.search(r'associated with ([^\.]+)', prompt)
            entity = entity_match.group(1) if entity_match else "the subject"
            
            # Generate a research strategy
            strategy = llm.generate_research_strategy(prompt, entity)
            
            # Convert the strategy to a string response
            import json
            return json.dumps(strategy)
            
        except ImportError:
            logger.warning("LLM integration module not found. Using simulated response.")
            return self._generate_simulated_response(prompt)
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return self._generate_simulated_response(prompt)
    
    def _generate_simulated_response(self, prompt: str) -> str:
        """
        Generate a simulated LLM response for testing.
        
        Args:
            prompt: The input prompt
            
        Returns:
            A simulated response text
        """
        # Extract entity from prompt
        entity_match = re.search(r'associated with ([^\.]+)', prompt)
        entity = entity_match.group(1) if entity_match else "the subject"
        
        # Generic response that doesn't hardcode crypto references
        return f"""
To research information associated with {entity}, I recommend checking the following sources:

Specific Sources to Check:
1. Personal website or blog if available
2. Social media profiles (Twitter, Instagram, LinkedIn)
3. Official organization websites if the entity is associated with any
4. News articles and interviews
5. Internet Archive for historical website versions
6. Academic or professional databases related to their field
7. GitHub or other code repositories if they're a developer or in tech
8. Public records relevant to their activities

Search Queries:
1. "{entity} biography"
2. "{entity} background"
3. "{entity} career history" 
4. "{entity} early work"
5. "{entity} interview"
6. "{entity} social media profiles"
7. "{entity} official website"
8. "{entity} publications OR projects"

These sources and queries should help you gather relevant information without making assumptions about the entity's specific field or activities.
"""
    
    def generate_research_strategy(self, objective: str, entity: str) -> Dict[str, Any]:
        """
        Generate a research strategy for a given objective and entity.
        
        Args:
            objective: The research objective
            entity: The target entity
            
        Returns:
            A dictionary containing research strategy details
        """
        # Check cache first
        cache_key = f"{objective}:{entity}"
        if cache_key in self.strategy_cache:
            logger.info(f"Using cached strategy for {objective} about {entity}")
            return self.strategy_cache[cache_key]
        
        # Determine the artifact type from the objective
        artifact_type = None
        for potential_type in ["name", "wallet", "code", "personal", "legal", "academic", "hidden", "institutional"]:
            if potential_type in objective.lower():
                artifact_type = potential_type
                break
        
        if not artifact_type:
            artifact_type = "general"
            logger.warning(f"Could not determine artifact type from objective: {objective}")
        
        # Get the template for this artifact type
        template = self.strategy_templates.get(artifact_type, {})
        
        # Generate the prompt
        if "prompt" in template:
            prompt = template["prompt"].format(entity=entity, current_project="", year="")
        else:
            prompt = f"I'm researching information associated with {entity}. What specific sources should I check? What specific search queries would help me find information about their early career, projects, or contributions?"
        
        # Get LLM response
        llm_response = self._call_llm_api(prompt)
        
        # Extract specific information from the LLM response
        sources = []
        search_queries = []
        github_repos = []
        usernames = []
        
        # Extract sources
        source_lines = re.findall(r'(?:https?://)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z0-9][-a-zA-Z0-9]*)+)(?:/[^\s]*)?', llm_response)
        sources.extend([f"https://{source}" if not source.startswith(('http://', 'https://')) else source for source in source_lines])
        
        # Extract search queries - lines that are in quotes
        search_query_lines = re.findall(r'"([^"]+)"', llm_response)
        search_queries.extend(search_query_lines)
        
        # Extract GitHub repos
        github_repos_lines = re.findall(r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)', llm_response)
        github_repos.extend([f"https://github.com/{repo}" for repo in github_repos_lines])
        
        # Extract usernames
        username_lines = re.findall(r'@([a-zA-Z0-9_-]+)', llm_response)
        usernames.extend(username_lines)
        
        # No hardcoded templates - use only LLM strategy
        
        # Prepare the result
        strategy = {
            "objective": objective,
            "entity": entity,
            "artifact_type": artifact_type,
            "llm_response": llm_response,
            "sources": sources,
            "search_queries": search_queries,
            "github_repos": github_repos,
            "usernames": usernames,
            "crawlable_urls": [],  # Will be populated below
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate crawlable URLs
        crawlable_urls = []
        
        # Normalize and add direct sources
        normalized_sources = []
        for source in sources:
            if not source.startswith(('http://', 'https://')):
                normalized_sources.append(f"https://{source}")
            else:
                normalized_sources.append(source)
        crawlable_urls.extend(normalized_sources)
        
        # Normalize and add GitHub repositories
        normalized_repos = []
        for repo in github_repos:
            if not repo.startswith(('http://', 'https://')):
                normalized_repos.append(f"https://{repo}")
            else:
                normalized_repos.append(repo)
        crawlable_urls.extend(normalized_repos)
        
        # Add search engine queries - only use DuckDuckGo, no GitHub code search
        for query in search_queries:
            encoded_query = quote_plus(query)
            crawlable_urls.append(f"https://duckduckgo.com/html/?q={encoded_query}")
        
        # Add Twitter searches - but only if they came from the LLM response
        if usernames:
            for username in usernames:
                crawlable_urls.append(f"https://twitter.com/{username}")
        
        # Removed hardcoded fallback - use LLM strategy only
        # Wayback machine URLs should come from the LLM strategy instead of hardcoded domains
        
        # Deduplicate URLs
        strategy["crawlable_urls"] = list(set(crawlable_urls))
        
        # Cache the strategy
        self.strategy_cache[cache_key] = strategy
        
        return strategy

if __name__ == "__main__":
    # Test the strategy generator
    strategy_generator = LLMResearchStrategy()
    
    test_objective = "Find name artifacts around Vitalik Buterin"
    test_entity = "Vitalik Buterin"
    
    strategy = strategy_generator.generate_research_strategy(test_objective, test_entity)
    
    print(f"Generated Research Strategy for: {test_objective}")
    print(f"Found {len(strategy['crawlable_urls'])} crawlable URLs")
    print(f"Search Queries: {len(strategy['search_queries'])}")
    print("\nSample URLs:")
    for url in strategy['crawlable_urls'][:5]:
        print(f"- {url}")
    
    print("\nSample Search Queries:")
    for query in strategy['search_queries'][:5]:
        print(f"- {query}")