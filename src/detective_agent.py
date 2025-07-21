#!/usr/bin/env python3
"""
Detective Agent for Narrahunt Phase 2.

This module implements an autonomous research agent that follows leads
and discovers artifacts related to narrative objectives. It acts as the
central reasoning engine that coordinates between different components.

Enhanced with Focused Exploration Strategy for methodical, depth-first investigations.
"""

import os
import sys

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(script_dir)
print(f"Changed working directory to: {script_dir}")

import os
import re
import json
import time
import logging
import random
import sys
import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urlparse, urljoin

# Set debug level temporarily
logging.getLogger('enhanced_artifact_detector').setLevel(logging.DEBUG)
logging.getLogger('detective_agent').setLevel(logging.DEBUG)
logging.getLogger('name_artifact_extractor').setLevel(logging.DEBUG)
logging.getLogger('research_strategy').setLevel(logging.DEBUG)

# Import internal modules
from core.narrative_matrix import NarrativeMatrix
from core.llm_integration import LLMIntegration
from core.crawler import Crawler
from core.wayback_integration import WaybackMachine
from core.enhanced_artifact_detector import EnhancedArtifactDetector
# Initialize the detector once to avoid multiple initializations
artifact_detector = EnhancedArtifactDetector()
from config.config_loader import get_api_key
from core.fetch import fetch_page  # Import fetch_page directly
from core.research_strategy import ResearchStrategy

# Configure logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_dir, 'logs', 'detective.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('detective_agent')

class DetectiveAgent:
    """
    Autonomous research agent that follows leads and discovers artifacts.
    
    This agent acts as a detective, following breadcrumbs and investigating
    leads to uncover information relevant to the research objective.
    """
    
    def __init__(self, objective: str, entity: str, max_iterations: int = 50, 
                 max_time_hours: float = 24.0, max_idle_iterations: int = 5):
        """
        Initialize the detective agent.
        
        Args:
            objective: The research objective (e.g., "Find name artifacts around Vitalik Buterin")
            entity: The primary entity to investigate (e.g., "Vitalik Buterin")
            max_iterations: Maximum number of investigation iterations
            max_time_hours: Maximum runtime in hours
            max_idle_iterations: Maximum number of iterations without new discoveries
        """
        self.objective = objective
        self.entity = entity
        self.max_iterations = max_iterations
        self.max_time_seconds = max_time_hours * 3600
        self.max_idle_iterations = max_idle_iterations
        
        # Initialize components
        self.narrative_matrix = NarrativeMatrix()
        self.llm = LLMIntegration(use_claude=True)
        self.llm_calls_count = 0
        self.crawler = Crawler()
        self.wayback = WaybackMachine()
        
        # Initialize focused research strategy
        self.research_strategy = ResearchStrategy()
        
        # Research state
        self.research_queue = []  # Legacy attribute, kept for compatibility
        self.investigated_urls = set()
        self.discoveries = []
        self.iteration_discoveries = {}
        self.current_iteration = 0
        self.idle_iterations = 0
        self.start_time = time.time()
        self.current_target = None  # Currently being investigated target
        
        # Research metadata
        self.investigation_strategies = {}
        
        # Initialize entity aliases with variations of the entity name
        self.entity_aliases = set([entity])
        if entity:
            # Add variations (first name, last name)
            name_parts = entity.split()
            for part in name_parts:
                if len(part) > 2:  # Only add name parts that are reasonably long
                    self.entity_aliases.add(part)
            
            # Add common forms of the entity name
            # If entity is "John Smith", also exclude "smith, john" etc.
            if len(name_parts) > 1:
                self.entity_aliases.add(f"{name_parts[-1]}, {' '.join(name_parts[:-1])}")
        
        # Discovery tracking for efficient deduplication
        self.unique_discovery_contents = set()
        
        # Initialize excluded names for auto-filtering
        self.excluded_names = self.entity_aliases.copy()
        
        # Initialize empty set for priority domains (will be populated by LLM strategy)
        self.priority_domains = set()
        
        self.investigation_history = []
        
        # Create results directory
        self.results_dir = os.path.join(base_dir, 'results', 'detective')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize research log
        self.research_log_path = os.path.join(self.results_dir, f'research_log_{int(time.time())}.jsonl')
        
        logger.info(f"Detective Agent initialized with objective: {objective}")
        logger.info(f"Primary entity: {entity}")
    
    def start_investigation(self):
        """Start the investigation process using focused, depth-first exploration."""
        logger.info("Starting investigation with focused exploration strategy...")
        
        # Initialize comprehensive research strategy
        self._initialize_research()
        
        # Check if strategy has targets after initialization
        strategy_status = self.research_strategy.get_status()
        if strategy_status['todo_count'] == 0:
            logger.error("Research strategy has no targets after initialization. Cannot proceed with investigation.")
            return []
            
        # Main investigation loop
        while self._should_continue_investigation():
            self.current_iteration += 1
            logger.info(f"\n{'='*80}\nStarting iteration {self.current_iteration}/{self.max_iterations}\n{'='*80}")
            
            # Check if we need to process any discovered leads
            if self.current_iteration % 5 == 0:  # Every 5 iterations
                self.research_strategy.process_discovered_leads()
                
            # Check if we need a strategy review
            if self.current_iteration % 10 == 0:  # Every 10 iterations
                self._review_research_strategy()
            
            # Get next investigation target using focused strategy
            target = self.research_strategy.get_next_target()
            if not target:
                logger.warning("No more targets to investigate in strategy")
                
                # Perform comprehensive strategy review with LLM
                self._review_research_strategy(force_update=True)
                
                # If still no targets after strategy review, exit
                if self.research_strategy.get_status()['todo_count'] == 0:
                    logger.info("No targets in research strategy after review. Ending investigation.")
                    break
                    
                continue
            
            # Set as current target
            self.current_target = target
            
            # Execute the investigation thoroughly
            new_discoveries = self._execute_investigation(target)
            
            # Track discoveries for this iteration
            self.iteration_discoveries[self.current_iteration] = new_discoveries
            
            # Mark current target as complete in strategy
            self.research_strategy.mark_target_complete(target, new_discoveries)
            
            # Reset idle iterations if we found something
            if new_discoveries:
                self.idle_iterations = 0
                
                # Instead of immediately consulting LLM, add these as leads to be investigated later
                for discovery in new_discoveries:
                    # Check if discovery contains potential leads
                    potential_leads = self._extract_leads_from_discovery(discovery)
                    for lead in potential_leads:
                        # Add to discovered leads, don't pursue immediately
                        self.research_strategy.add_discovered_lead(lead)
                
                # Log the investigation results
                self._log_investigation_results(target, new_discoveries, [])
            else:
                # No new discoveries in this iteration
                logger.info(f"No new discoveries in iteration {self.current_iteration}, continuing with next target")
            
            # Save state after each iteration
            self._save_state()
        
        logger.info(f"Investigation completed after {self.current_iteration} iterations")
        logger.info(f"Total discoveries: {len(self.discoveries)}")
        
        # Generate final report
        self._generate_investigation_report()
        
        return self.discoveries
    
    def _get_llm_instance(self):
        """Alternate between Claude and GPT for each call."""
        use_claude = (self.llm_calls_count % 2 == 0)
        self.llm_calls_count += 1
        
        llm_type = "Claude" if use_claude else "GPT"
        logger.info(f"Using {llm_type} for LLM call #{self.llm_calls_count}")
        
        return LLMIntegration(use_claude=use_claude, use_openai=not use_claude)

    def _initialize_research(self):
        """Initialize the research state with a comprehensive strategy."""
        logger.info("Initializing comprehensive research strategy...")
        
        # Get initial research strategy from LLM - now a comprehensive TODO list
        initial_strategy = self._get_initial_research_strategy()
        
        # If strategy is empty, FAIL IMMEDIATELY:
        if not initial_strategy:
            logger.error("No strategy received from LLM - investigation cannot proceed")
            return  # Exit immediately
            
        # Extract initial targets from the strategy
        initial_targets = []
        
        # Add websites to check
        for source in initial_strategy.get('sources', []):
            # Extract URL from source dictionary or use directly if it's a string
            url = source.get('url') if isinstance(source, dict) else source
            
            if self._is_valid_url(url):
                # Assign different priorities based on domain relevance
                domain = self._extract_domain(url)
                
                # Use priority from the source if available, otherwise default
                priority = source.get('priority', 10) if isinstance(source, dict) else 10
                priority = priority if domain not in self.priority_domains else 10
                
                # Get rationale if available
                rationale = source.get('rationale', 'Initial source from comprehensive research strategy') if isinstance(source, dict) else 'Initial source from comprehensive research strategy'
                
                initial_targets.append({
                    'url': url,
                    'type': 'website',
                    'priority': priority,
                    'rationale': rationale,
                    'use_wayback': True
                })
            else:
                # If it's not a URL but a source name, add it to prioritized domains
                domain = self._extract_domain(url) if url else None
                if domain:
                    self.priority_domains.add(domain)
        
        # Add GitHub repositories with high priority
        for repo in initial_strategy.get('github_targets', []):
            # Extract URL from repo dictionary or use directly if it's a string
            url = repo.get('url') if isinstance(repo, dict) else repo
            
            if self._is_valid_url(url):
                # Use priority and rationale from the repo if available, otherwise default
                priority = repo.get('priority', 9) if isinstance(repo, dict) else 9
                rationale = repo.get('rationale', 'GitHub repository from comprehensive research strategy') if isinstance(repo, dict) else 'GitHub repository from comprehensive research strategy'
                
                initial_targets.append({
                    'url': url,
                    'type': 'github',
                    'priority': priority,
                    'rationale': rationale
                })
        
        # Add wayback targets for important historical periods
        for wayback_target in initial_strategy.get('wayback_targets', []):
            # Extract URL from wayback_target dictionary or use directly if it's a string
            url = wayback_target.get('url') if isinstance(wayback_target, dict) else wayback_target
            
            if self._is_valid_url(url):
                # Use priority and rationale from the wayback_target if available, otherwise default
                priority = wayback_target.get('priority', 7) if isinstance(wayback_target, dict) else 7
                rationale = wayback_target.get('rationale', 'Historical investigation from comprehensive research strategy') if isinstance(wayback_target, dict) else 'Historical investigation from comprehensive research strategy'
                
                # Extract year range if available
                if isinstance(wayback_target, dict) and 'years' in wayback_target:
                    years = wayback_target['years']
                    # If years is a list with at least 2 elements, use as range
                    if isinstance(years, list) and len(years) >= 2:
                        year_range = (min(years), max(years))
                    else:
                        year_range = (2013, datetime.datetime.now().year)
                else:
                    year_range = (2013, datetime.datetime.now().year)
                
                initial_targets.append({
                    'url': url,
                    'type': 'wayback',
                    'priority': priority,
                    'rationale': rationale,
                    'year_range': year_range
                })
        
        # Add search queries with various priorities
        for search_item in initial_strategy.get('search_queries', []):
            # Extract query from search_item dictionary or use directly if it's a string
            query = search_item.get('query') if isinstance(search_item, dict) else search_item
            
            if query:
                # Use priority and rationale from the search_item if available, otherwise default
                priority = search_item.get('priority', 8) if isinstance(search_item, dict) else 8
                rationale = search_item.get('rationale', 'Search query from comprehensive research strategy') if isinstance(search_item, dict) else 'Search query from comprehensive research strategy'
                
                initial_targets.append({
                    'query': query,
                    'type': 'search',
                    'priority': priority,
                    'rationale': rationale,
                    'engine': 'google'
                })
        
        # Add all targets to the research strategy
        logger.info(f"Adding {len(initial_targets)} initial targets to research strategy")
        self.research_strategy.add_targets(initial_targets)
        
        # For backward compatibility, also update the legacy research queue
        self._update_research_queue(initial_targets)
        
        # Log the initialization
        self._log_to_research_log({
            'event': 'initialization',
            'timestamp': self._get_timestamp(),
            'objective': self.objective,
            'entity': self.entity,
            'initial_strategy': initial_strategy,
            'initial_targets': initial_targets
        })
    
    def _get_initial_research_strategy(self) -> Dict[str, Any]:
        """Get the comprehensive initial research strategy from the LLM."""
        logger.info("Getting comprehensive initial research strategy from LLM...")
        
        # Custom prompt for generating a comprehensive research TODO list
        prompt = f"""GENERATE RESEARCH STRATEGY JSON NOW.

TARGET: {self.entity}

OUTPUT FORMAT - EXACT JSON ONLY:
{{
    "sources": [],
    "search_queries": [],
    "information_types": ["usernames", "project_names", "handles"],
    "time_periods": []
}}

RULES:
- Generate 3-5 items per array
- Use actual search terms and URLs specific to the target entity
- Focus on finding old usernames, project names, wallet addresses
- Target historical crypto forums, early GitHub repos, archived sites
- Do NOT use generic crypto URLs - all URLs must be specific to the target entity

NO EXPLANATION. NO OTHER TEXT. ONLY JSON."""
        
        # Call the LLM with alternating models
        try:
            # Try Claude first
            llm = self._get_llm_instance()
            response = None
            
            if llm.use_claude:
                response = llm._call_claude(prompt)
                logger.debug(f"RAW LLM RESPONSE: {response[:500]}...")
                
            # If Claude failed, try OpenAI
            if not response or response.strip() == "{}":
                logger.warning("Claude failed, trying OpenAI as backup")
                try:
                    backup_llm = LLMIntegration(use_claude=False, use_openai=True)
                    response = backup_llm._call_openai(prompt)
                    logger.debug(f"RAW LLM RESPONSE: {response[:500]}...")
                except Exception as e:
                    logger.warning(f"OpenAI also failed: {str(e)}")
                    response = None
                    
            # If both Claude and OpenAI failed, try Gemini
            if not response or response.strip() == "{}":
                logger.warning("Claude and OpenAI failed, trying Gemini as final backup")
                try:
                    gemini_llm = LLMIntegration(use_claude=False, use_openai=False, use_gemini=True)
                    response = gemini_llm._call_gemini(prompt)
                    logger.debug(f"RAW LLM RESPONSE: {response[:500]}...")
                except Exception as e:
                    logger.error(f"All LLM services failed. Gemini error: {str(e)}")
                    response = "{}"
            
            # Log the raw response for debugging
            logger.info(f"RAW LLM RESPONSE: {response}")
            
            # Extract JSON from the response
            json_str = llm._extract_json(response)
            logger.debug(f"EXTRACTED JSON: {json_str}")
            data = json.loads(json_str)
            
            # More comprehensive strategy with detailed information for each target
            strategy = {
                "sources": data.get("sources", []),
                "github_targets": data.get("github_targets", []),
                "wayback_targets": data.get("wayback_targets", []),
                "search_queries": data.get("search_queries", []),
                "forum_targets": data.get("forum_targets", []),
                "key_time_periods": data.get("key_time_periods", [])
            }
            
            # For backward compatibility, convert the more detailed items to simple lists
            # This is only used for logging and validation
            simple_strategy = {
                "sources": [s.get('url') if isinstance(s, dict) else s for s in data.get("sources", [])],
                "search_queries": [q.get('query') if isinstance(q, dict) else q for q in data.get("search_queries", [])],
                "github_targets": [g.get('url') if isinstance(g, dict) else g for g in data.get("github_targets", [])],
                "wayback_targets": [w.get('url') if isinstance(w, dict) else w for w in data.get("wayback_targets", [])]
            }
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Raw LLM response: {mock_result}")
            strategy = {
                "sources": [],
                "github_targets": [],
                "wayback_targets": [],
                "search_queries": [],
                "forum_targets": [],
                "key_time_periods": []
            }
            simple_strategy = {
                "sources": [],
                "search_queries": [],
                "github_targets": [],
                "wayback_targets": []
            }
        
        # Log the strategy
        logger.info(f"Comprehensive research strategy obtained:")
        logger.info(f"Sources: {len(strategy.get('sources', []))} items")
        logger.info(f"GitHub targets: {len(strategy.get('github_targets', []))} items")
        logger.info(f"Wayback targets: {len(strategy.get('wayback_targets', []))} items")
        logger.info(f"Search queries: {len(strategy.get('search_queries', []))} items")
        logger.info(f"Forum targets: {len(strategy.get('forum_targets', []))} items")
        logger.info(f"Key time periods: {len(strategy.get('key_time_periods', []))} items")
        
        # If strategy is empty or missing critical components, log error and raise exception
        if (not strategy.get('sources') and 
            not strategy.get('search_queries') and 
            not strategy.get('github_targets') and
            not strategy.get('wayback_targets')):
            logger.error("LLM strategy generation failed - no targets to investigate")
            # Raise exception instead of returning empty dictionary
            raise Exception("LLM strategy generation failed - cannot proceed without valid strategy")
        
        return strategy
    
    def _execute_investigation(self, target: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute an investigation on a target, implementing proper depth-first exploration.
        
        This method ensures complete exploration of a domain before moving to the next one,
        by extracting all links from the current domain and adding them to the queue with
        the same priority as the current target.
        
        Args:
            target: The investigation target
            
        Returns:
            List of new discoveries
        """
        target_type = target.get('type', 'unknown')
        
        # Set current focus target for depth-first exploration
        if target_type in ['website', 'github', 'wayback']:
            domain = self._extract_domain(target.get('url', ''))
            if domain and not hasattr(self, 'current_focus_target'):
                self.current_focus_target = domain
                logger.info(f"Setting current focus target to: {self.current_focus_target}")
            elif domain and self.current_focus_target != domain:
                # Only switch focus if the current domain is exhausted
                if self._is_target_exhausted(self.current_focus_target):
                    logger.info(f"Switching focus target from {self.current_focus_target} to {domain}")
                    self.current_focus_target = domain
        
        # Track the target as partially explored until we've exhausted all related links
        target['exploration_status'] = 'partially_explored'
        
        # Execute the appropriate investigation method based on target type
        discoveries = []
        if target_type == 'website':
            discoveries = self._investigate_website(target)
        elif target_type == 'search':
            discoveries = self._execute_search(target)
        elif target_type == 'wayback':
            discoveries = self._investigate_wayback(target)
        elif target_type == 'github':
            discoveries = self._investigate_github(target)
        else:
            logger.warning(f"Unknown target type: {target_type}")
            return []
            
        # Check if this domain is now fully exhausted
        if target_type in ['website', 'github', 'wayback']:
            domain = self._extract_domain(target.get('url', ''))
            if domain and self._is_target_exhausted(domain):
                logger.info(f"Domain {domain} is now fully explored")
                # Update all targets from this domain to mark them as fully explored
                for completed_target in self.research_strategy.completed_targets:
                    if completed_target.get('type') in ['website', 'github', 'wayback']:
                        if self._extract_domain(completed_target.get('url', '')) == domain:
                            completed_target['exploration_status'] = 'fully_explored'
                
                # If this was our current focus, we need to choose a new focus
                if self.current_focus_target == domain:
                    self.current_focus_target = None
                    
        return discoveries
    
    def _is_target_exhausted(self, domain: str) -> bool:
        """
        Check if a target domain has been fully explored (no more URLs from that domain in queue).
        
        Args:
            domain: The domain name to check
            
        Returns:
            True if the domain is exhausted (no more URLs in queue), False otherwise
        """
        # Check if there are any URLs from this domain in the research queue
        for target in self.research_queue:
            if target.get('type') in ['website', 'github', 'wayback']:
                target_domain = self._extract_domain(target.get('url', ''))
                if target_domain == domain:
                    # Found at least one URL from this domain in queue
                    return False
        
        # No URLs from this domain in queue
        return True
        
    def _investigate_website(self, target: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Investigate a website target and extract all links from that domain for further investigation.
        
        This implements proper depth-first exploration by finding all links from the current domain
        and adding them to the queue with the same priority as the current target.
        
        Args:
            target: The website target to investigate
            
        Returns:
            List of discoveries from this website
        """
        url = target.get('url')
        
        if not url:
            logger.warning("Website target missing URL")
            return []
        
        # Special handling for Wayback Machine calendar URLs (containing *)
        if '*' in url and 'web.archive.org/web/' in url:
            logger.info(f"Detected Wayback calendar URL: {url}")
            return self._investigate_wayback_calendar(url)
        
        if url in self.investigated_urls:
            logger.info(f"URL already investigated: {url}")
            return []
        
        logger.info(f"Investigating website: {url}")
        self.investigated_urls.add(url)
        
        # Get the domain for focused exploration
        target_domain = self._extract_domain(url)
        
        # Crawl the website
        try:
            # Use fetch_page directly instead of self.crawler.fetch_url
            html_content, response_info = fetch_page(url)
            final_url = response_info.get('final_url', url) if response_info else url
            
            if not html_content:
                logger.warning(f"Failed to fetch content from {url}")
                
                # If fetch failed, try Wayback Machine if enabled
                if target.get('use_wayback', False):
                    logger.info(f"Trying Wayback Machine for {url}")
                    return self._investigate_wayback({'url': url, 'type': 'wayback'})
                return []
            
            # Extract artifacts from the content
            logger.info(f"Extracting artifacts from {len(html_content)} bytes of content from {final_url}")
            artifacts = artifact_detector.extract_artifacts(
                html_content, 
                final_url, 
                objective=self.objective,
                entity=self.entity
            )
            logger.info(f"Found {len(artifacts)} artifacts from {final_url}")
            
            # Process the artifacts into discoveries
            discoveries = self._process_artifacts(artifacts, url)
            
            # Determine site type and perform deep investigation if needed
            site_type = self._detect_site_type(final_url, html_content)
            logger.info(f"Detected site type for {final_url}: {site_type}")
            
            if site_type == 'forum':
                logger.info(f"Performing deep forum investigation for {final_url}")
                deep_discoveries = self._investigate_forum_deeply(final_url, html_content)
                if deep_discoveries:
                    logger.info(f"Found {len(deep_discoveries)} additional discoveries from deep forum investigation")
                    discoveries.extend(deep_discoveries)
            elif site_type == 'blog':
                logger.info(f"Performing deep blog investigation for {final_url}")
                deep_discoveries = self._investigate_blog_deeply(final_url, html_content)
                if deep_discoveries:
                    logger.info(f"Found {len(deep_discoveries)} additional discoveries from deep blog investigation")
                    discoveries.extend(deep_discoveries)
            
            # Extract links from the page for depth-first exploration using improved deduplication
            new_targets = []
            current_priority = target.get('priority', 5)
            
            # Use our improved link extraction method with built-in deduplication
            same_domain_links = self._extract_same_domain_links(html_content, final_url)
            
            # Create targets from the deduplicated links
            for full_url in same_domain_links:
                # Create unique target ID for deduplication in queue
                target_id = f"website:{self._normalize_url_for_deduplication(full_url)}"
                
                new_targets.append({
                    'url': full_url,
                    'type': 'website',
                    'id': target_id,  # Add ID for queue deduplication
                    'priority': current_priority,
                    'rationale': f'Same-domain link from {url} - depth-first exploration',
                    'use_wayback': target.get('use_wayback', False)
                })
            
            logger.info(f"Found {len(same_domain_links)} new links from domain {self._extract_domain(final_url)}")
            
            # Add the new targets to the research queue
            if new_targets:
                self._update_research_queue(new_targets)
            
            # If use_wayback is enabled, also check historical versions
            if target.get('use_wayback', False) and not url.startswith('https://web.archive.org/'):
                # Add wayback investigation to the queue with lower priority
                self._update_research_queue([{
                    'url': url,
                    'type': 'wayback',
                    'priority': current_priority - 2,  # Lower priority than direct links
                    'rationale': f'Historical investigation of {url}',
                    'year_range': target.get('year_range', (2013, datetime.datetime.now().year))
                }])
            
            return discoveries
        
        except Exception as e:
            logger.error(f"Error investigating website {url}: {str(e)}")
            return []
    
    def _investigate_wayback_calendar(self, calendar_url: str) -> List[Dict[str, Any]]:
        """
        Investigate a Wayback Machine calendar URL (with * wildcard).
        
        Args:
            calendar_url: The Wayback Machine calendar URL
        
        Returns:
            List of discoveries
        """
        # Extract the timestamp and original URL from the calendar URL
        # Various possible formats:
        # - https://web.archive.org/web/YYYYMMDDHHMMSS*/https://original.url
        # - https://web.archive.org/web/YYYY*/https://original.url
        
        # Try to match with full timestamp format first
        match = re.match(r'https://web\.archive\.org/web/(\d{4,14})\*/(.+)', calendar_url)
        if not match:
            logger.warning(f"Invalid Wayback calendar URL format: {calendar_url}")
            return []
        
        timestamp = match.group(1)
        original_url = match.group(2)
        
        # Extract year from the timestamp (first 4 digits)
        year = timestamp[:4]
        
        logger.info(f"Processing Wayback calendar for {original_url} from year {year}")
        
        # Convert the Wayback calendar URL to a direct snapshot URL
        # This ensures we're looking at actual archived content
        wayback_url = f"https://web.archive.org/web/{timestamp}/http://{original_url.replace('https://', '').replace('http://', '')}"
        
        logger.info(f"Converted calendar URL to direct snapshot URL: {wayback_url}")
        
        # Directly investigate the wayback URL
        try:
            html_content, response_info = fetch_page(wayback_url)
            
            if not html_content:
                logger.warning(f"Failed to fetch content from direct Wayback URL: {wayback_url}")
                
                # Fall back to year-based search if direct URL fails
                return self._investigate_wayback({
                    'url': original_url,
                    'type': 'wayback',
                    'year_range': (int(year), int(year))
                })
            
            logger.info(f"Extracting artifacts from {len(html_content)} bytes of Wayback content from {wayback_url}")
            artifacts = artifact_detector.extract_artifacts(
                html_content, 
                wayback_url, 
                date=timestamp,
                objective=self.objective,
                entity=self.entity
            )
            logger.info(f"Found {len(artifacts)} artifacts from Wayback snapshot {wayback_url}")
            
            # Process the artifacts into discoveries
            return self._process_artifacts(artifacts, wayback_url, is_wayback=True, original_url=original_url)
            
        except Exception as e:
            logger.error(f"Error investigating direct Wayback URL {wayback_url}: {str(e)}")
            
            # Fall back to year-based search if direct URL fails
            return self._investigate_wayback({
                'url': original_url,
                'type': 'wayback',
                'year_range': (int(year), int(year))
            })
    
    def _investigate_wayback(self, target: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Investigate a URL using the Wayback Machine."""
        url = target.get('url')
        
        if not url:
            logger.warning("Wayback target missing URL")
            return []
        
        wayback_url = f"wayback:{url}"
        if wayback_url in self.investigated_urls:
            logger.info(f"Wayback URL already investigated: {url}")
            return []
        
        logger.info(f"Investigating historical versions of: {url}")
        self.investigated_urls.add(wayback_url)
        
        # Get year range
        year_range = target.get('year_range', (2013, datetime.datetime.now().year))
        
        # Get snapshots from Wayback Machine
        try:
            # Convert years to strings for the API
            from_date = str(year_range[0])
            to_date = str(year_range[1])
            snapshots = self.wayback.get_snapshots(url, from_date=from_date, to_date=to_date)
            
            if not snapshots:
                logger.warning(f"No Wayback Machine snapshots found for {url}")
                return []
            
            logger.info(f"Found {len(snapshots)} Wayback Machine snapshots for {url}")
            
            # Sample snapshots if there are too many
            if len(snapshots) > 5:
                # Get earliest, latest, and 3 random snapshots in between
                sorted_snapshots = sorted(snapshots, key=lambda x: x.get('timestamp', ''))
                selected_snapshots = [
                    sorted_snapshots[0],  # Earliest
                    sorted_snapshots[-1]  # Latest
                ]
                
                # Add 3 random snapshots if available
                if len(sorted_snapshots) > 2:
                    middle_snapshots = sorted_snapshots[1:-1]
                    random_samples = random.sample(middle_snapshots, min(3, len(middle_snapshots)))
                    selected_snapshots.extend(random_samples)
            else:
                selected_snapshots = snapshots
            
            all_discoveries = []
            
            # Investigate each selected snapshot
            for snapshot in selected_snapshots:
                # Check for 'wayback_url' which is the field used in wayback_integration.py
                snapshot_url = snapshot.get('wayback_url')
                if not snapshot_url:
                    logger.warning(f"Missing wayback_url in snapshot: {snapshot}")
                    continue
                
                logger.info(f"Investigating Wayback snapshot: {snapshot_url}")
                
                try:
                    # Use fetch_page directly instead of self.crawler.fetch_url
                    html_content, response_info = fetch_page(snapshot_url)
                    
                    if not html_content:
                        logger.warning(f"Failed to fetch content from Wayback snapshot: {snapshot_url}")
                        continue
                    
                    # Extract artifacts from the content
                    logger.info(f"Extracting artifacts from {len(html_content)} bytes of Wayback content from {snapshot_url}")
                    artifacts = artifact_detector.extract_artifacts(
                        html_content, 
                        snapshot_url, 
                        date=snapshot.get('timestamp'),
                        objective=self.objective,
                        entity=self.entity
                    )
                    logger.info(f"Found {len(artifacts)} artifacts from Wayback snapshot {snapshot_url}")
                    
                    # Process the artifacts into discoveries
                    discoveries = self._process_artifacts(artifacts, snapshot_url, is_wayback=True, original_url=url)
                    all_discoveries.extend(discoveries)
                    
                except Exception as e:
                    logger.error(f"Error investigating Wayback snapshot {snapshot_url}: {str(e)}")
            
            return all_discoveries
        
        except Exception as e:
            logger.error(f"Error investigating Wayback for {url}: {str(e)}")
            return []
    
    def _execute_search(self, target: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a search query and process the results."""
        query = target.get('query')
        
        if not query:
            logger.warning("Search target missing query")
            return []
        
        search_key = f"search:{query}"
        if search_key in self.investigated_urls:
            logger.info(f"Search query already executed: {query}")
            return []
        
        logger.info(f"Executing search query: {query}")
        self.investigated_urls.add(search_key)
        
        # Execute the search
        try:
            # Convert query string to a list of keywords
            keywords = query.split()
            search_results = self.crawler.search(keywords, max_results=10)
            
            if not search_results:
                logger.warning(f"No search results found for query: {query}")
                return []
            
            logger.info(f"Found {len(search_results)} search results for query: {query}")
            
            # Add search results to the research queue
            search_targets = []
            for result in search_results[:10]:  # Limit to top 10 results
                result_url = result.get('url')
                if not result_url or result_url in self.investigated_urls:
                    continue
                
                search_targets.append({
                    'url': result_url,
                    'type': 'website',
                    'priority': target.get('priority', 5) - 1,
                    'rationale': f'Search result for "{query}"',
                    'use_wayback': self._should_check_wayback(result_url)
                })
            
            self._update_research_queue(search_targets)
            
            # Return empty discoveries list since we just added targets to the queue
            return []
        
        except Exception as e:
            logger.error(f"Error executing search for {query}: {str(e)}")
            return []
    
    def _extract_github_repo_info(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse GitHub URLs to extract owner and repository name.
        
        Args:
            url: GitHub URL to parse
            
        Returns:
            Tuple of (owner, repository) or (None, None) if not a valid repository URL
        """
        # Try to extract owner and repo from the URL
        github_repo_pattern = r'github\.com[/:]([^/]+)/([^/]+)'
        match = re.search(github_repo_pattern, url)
        
        if not match:
            logger.warning(f"Could not extract repository info from GitHub URL: {url}")
            return None, None
            
        owner = match.group(1)
        repo = match.group(2)
        
        # Clean up the repository name (remove .git suffix if present)
        if repo.endswith('.git'):
            repo = repo[:-4]
            
        # Handle potential query parameters or trailing slashes
        if '?' in repo:
            repo = repo.split('?')[0]
        if '#' in repo:
            repo = repo.split('#')[0]
            
        logger.info(f"Extracted GitHub repository: {owner}/{repo}")
        return owner, repo
        
    def _investigate_github_files(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        Investigate GitHub repository files for artifacts.
        
        Args:
            owner: Repository owner/organization
            repo: Repository name
            
        Returns:
            List of discoveries
        """
        discoveries = []
        
        # GitHub raw content base URL
        base_raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/master"
        base_raw_url_main = f"https://raw.githubusercontent.com/{owner}/{repo}/main"
        
        # Common files to check for name artifacts
        important_files = [
            'README.md',
            'README.rst',
            'CONTRIBUTORS.md',
            'CONTRIBUTORS',
            'AUTHORS',
            'AUTHORS.md',
            'package.json',
            'setup.py',
            'Cargo.toml',
            'pyproject.toml',
            'LICENSE',
            'CHANGELOG.md',
            'HISTORY.md',
            'docs/index.md',
            'docs/README.md',
            '.github/FUNDING.yml'
        ]
        
        # Try to fetch each important file
        for file_path in important_files:
            # Try main branch first (newer repositories)
            content = None
            url = f"{base_raw_url_main}/{file_path}"
            
            try:
                logger.info(f"Checking GitHub file: {url}")
                content, response_info = fetch_page(url)
                source_url = url
                
                if not content:
                    # Try master branch if main fails (older repositories)
                    url = f"{base_raw_url}/{file_path}"
                    logger.info(f"Checking GitHub file: {url}")
                    content, response_info = fetch_page(url)
                    source_url = url
                    
                    if not content:
                        logger.info(f"GitHub file not found: {file_path}")
                        continue  # Try next file instead of erroring
            except Exception as e:
                logger.error(f"Error fetching GitHub file {file_path}: {str(e)}")
                continue
                
            if content:
                # Extract artifacts from file content
                try:
                    file_type = file_path.split('.')[-1].lower() if '.' in file_path else 'txt'
                    
                    # Extract artifacts based on file type
                    if file_type in ['md', 'rst', 'txt']:
                        # Markdown/text files often contain project descriptions, contributor lists
                        artifacts = artifact_detector.extract_artifacts(content, context=f"GitHub {file_path}")
                    elif file_type in ['json', 'toml', 'py']:
                        # Configuration files often contain project metadata
                        artifacts = artifact_detector.extract_artifacts(content, context=f"GitHub config {file_path}")
                    else:
                        # General case
                        artifacts = artifact_detector.extract_artifacts(content, context=f"GitHub file {file_path}")
                    
                    # Process the artifacts
                    file_discoveries = self._process_artifacts(artifacts, source_url)
                    
                    # Add file-specific metadata
                    for discovery in file_discoveries:
                        discovery['github_file'] = file_path
                        discovery['github_repo'] = f"{owner}/{repo}"
                    
                    discoveries.extend(file_discoveries)
                    logger.info(f"Found {len(file_discoveries)} artifacts in {file_path}")
                except Exception as e:
                    logger.error(f"Error extracting artifacts from {file_path}: {str(e)}")
        
        # Fetch and analyze commits for contributor names and references
        self._investigate_github_commits(owner, repo, discoveries)
        
        # Fetch and analyze issues for discussions (more likely to contain name artifacts)
        self._investigate_github_issues(owner, repo, discoveries)
        
        return discoveries
        
    def _investigate_github_commits(self, owner: str, repo: str, discoveries: List[Dict[str, Any]]) -> None:
        """
        Investigate GitHub commit history for artifacts.
        
        Args:
            owner: Repository owner/organization
            repo: Repository name
            discoveries: List to append new discoveries to
        """
        # Commits URL
        commits_url = f"https://github.com/{owner}/{repo}/commits"
        
        try:
            logger.info(f"Fetching GitHub commits: {commits_url}")
            content, response_info = fetch_page(commits_url)
            
            if content:
                # Extract commit messages and authors from the HTML
                commit_pattern = r'<a class="commit-title.*?>(.*?)</a>.*?<a class="commit-author.*?>(.*?)</a>'
                matches = re.findall(commit_pattern, content, re.DOTALL)
                
                commit_text = "\n".join([f"Commit by {author.strip()}: {message.strip()}" 
                                        for message, author in matches])
                
                # Extract artifacts from commit history
                if commit_text:
                    artifacts = artifact_detector.extract_artifacts(
                        commit_text, context=f"GitHub commits for {owner}/{repo}")
                    
                    commit_discoveries = self._process_artifacts(artifacts, commits_url)
                    
                    # Add commit-specific metadata
                    for discovery in commit_discoveries:
                        discovery['github_source'] = 'commits'
                        discovery['github_repo'] = f"{owner}/{repo}"
                    
                    discoveries.extend(commit_discoveries)
                    logger.info(f"Found {len(commit_discoveries)} artifacts in commit history")
        except Exception as e:
            logger.error(f"Error investigating commits for {owner}/{repo}: {str(e)}")
    
    def _investigate_github_issues(self, owner: str, repo: str, discoveries: List[Dict[str, Any]]) -> None:
        """
        Investigate GitHub issues for artifacts.
        
        Args:
            owner: Repository owner/organization
            repo: Repository name
            discoveries: List to append new discoveries to
        """
        # Issues URL
        issues_url = f"https://github.com/{owner}/{repo}/issues"
        
        try:
            logger.info(f"Fetching GitHub issues: {issues_url}")
            content, response_info = fetch_page(issues_url)
            
            if content:
                # Extract issue titles and authors from the HTML
                issue_pattern = r'<a class="Link--primary v-align-middle.*?>(.*?)</a>.*?opened by.*?<a.*?>(.*?)</a>'
                matches = re.findall(issue_pattern, content, re.DOTALL)
                
                issue_text = "\n".join([f"Issue by {author.strip()}: {title.strip()}" 
                                      for title, author in matches])
                
                # Extract artifacts from issues
                if issue_text:
                    artifacts = artifact_detector.extract_artifacts(
                        issue_text, context=f"GitHub issues for {owner}/{repo}")
                    
                    issue_discoveries = self._process_artifacts(artifacts, issues_url)
                    
                    # Add issue-specific metadata
                    for discovery in issue_discoveries:
                        discovery['github_source'] = 'issues'
                        discovery['github_repo'] = f"{owner}/{repo}"
                    
                    discoveries.extend(issue_discoveries)
                    logger.info(f"Found {len(issue_discoveries)} artifacts in issues")
        except Exception as e:
            logger.error(f"Error investigating issues for {owner}/{repo}: {str(e)}")
    
    def _investigate_github(self, target: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Investigate a GitHub repository in depth.
        
        Args:
            target: Target dictionary containing the URL and other metadata
            
        Returns:
            List of discoveries
        """
        github_url = target.get('url')
        
        if not github_url:
            logger.warning("GitHub target missing URL")
            return []
        
        if github_url in self.investigated_urls:
            logger.info(f"GitHub URL already investigated: {github_url}")
            return []
        
        logger.info(f"Investigating GitHub repository: {github_url}")
        self.investigated_urls.add(github_url)
        
        # Extract owner and repo from the URL
        owner, repo = self._extract_github_repo_info(github_url)
        
        # If we couldn't extract owner/repo, fall back to treating it as a regular website
        if not owner or not repo:
            logger.warning(f"Could not parse GitHub URL: {github_url}. Treating as regular website.")
            return self._investigate_website({'url': github_url, 'type': 'website'})
        
        # Investigate the GitHub repository files
        discoveries = self._investigate_github_files(owner, repo)
        
        # Also investigate the main GitHub page for additional context
        website_discoveries = self._investigate_website({'url': github_url, 'type': 'website'})
        
        # Combine discoveries from both sources
        all_discoveries = discoveries + website_discoveries
        
        logger.info(f"Found {len(all_discoveries)} total artifacts in GitHub repository {owner}/{repo}")
        return all_discoveries
    
    def _process_artifacts(self, artifacts: List[Dict[str, Any]], source_url: str,
                          is_wayback: bool = False, original_url: str = None) -> List[Dict[str, Any]]:
        """
        Process artifacts into discoveries.
        
        Args:
            artifacts: List of artifacts from the artifact extractor
            source_url: The URL where the artifacts were found
            is_wayback: Whether this is from a Wayback Machine snapshot
            original_url: The original URL if is_wayback is True
            
        Returns:
            List of new discoveries
        """
        if not artifacts:
            return []
        
        # Log raw artifacts before processing
        logger.info(f"Raw artifacts from {source_url}: {len(artifacts)} total")
        for i, artifact in enumerate(artifacts):
            logger.debug(f"Raw artifact {i+1}: {artifact.get('summary', 'No summary')} ({artifact.get('type', 'unknown')})")
        
        # Track artifacts for deduplication within this batch
        batch_artifacts = {}
        new_discoveries = []
        skipped_count = 0
        
        for artifact in artifacts:
            artifact_type = artifact.get('type', 'unknown')
            content = artifact.get('content', '')
            name = artifact.get('name', '')
            
            # Skip empty content
            if not content and not name:
                skipped_count += 1
                continue
                
            # Normalize content for deduplication within this batch using type-specific normalization
            normalized_content = self._normalize_content(content, content_type=artifact_type)
            
            # For name artifacts, also normalize the name field
            normalized_name = None
            if artifact_type == 'name' and name:
                normalized_name = self._normalize_content(name, content_type='name')
            
            # Skip if we already processed this artifact in this batch - check both content and name
            if (normalized_content and normalized_content in batch_artifacts) or \
               (normalized_name and normalized_name in batch_artifacts):
                logger.info(f"Skipping duplicate artifact within batch: {content[:40]}...")
                skipped_count += 1
                continue
                
            # Track this artifact for batch deduplication
            if normalized_content:
                batch_artifacts[normalized_content] = True
            if normalized_name:
                batch_artifacts[normalized_name] = True
                
            # For name artifacts, also do an extra check against core name (no special chars)
            if artifact_type == 'name' and content:
                core_name = re.sub(r'[^\w]', '', content.lower())
                if core_name in batch_artifacts:
                    logger.info(f"Skipping name with same core within batch: {content}")
                    skipped_count += 1
                    continue
                if core_name:
                    batch_artifacts[core_name] = True
                
            # Skip low-scoring artifacts and prioritize high-value types
            score = artifact.get('score', 0)
            
            # For name artifacts, apply additional filtering and scoring
            if artifact_type == 'name':
                # Skip if content looks like a sentence fragment (with common prepositions/articles)
                words = content.split()
                if any(word.lower() in ['the', 'a', 'an', 'of', 'to', 'from', 'by', 'with', 'for', 'in', 'on', 'at'] 
                      for word in words[:1] + words[-1:]):  # Check first and last word
                    logger.debug(f"Skipping sentence fragment: {content}")
                    skipped_count += 1
                    continue
                
                # Skip if the name is part of the entity or in entity aliases
                if self.entity and (self.entity.lower() in content.lower() or 
                                   any(alias.lower() in content.lower() for alias in self.entity_aliases)):
                    logger.debug(f"Skipping entity-related name: {content}")
                    skipped_count += 1
                    continue
                
                # Additional checks for name quality
                if re.search(r'[\n\r\t]', content):  # Contains newlines or tabs
                    logger.debug(f"Skipping name with invalid characters: {content}")
                    skipped_count += 1
                    continue
                    
                # Skip names that are too short or too long
                if len(content) < 2 or len(content) > 30:
                    logger.debug(f"Skipping name with invalid length: {content}")
                    skipped_count += 1
                    continue
                
                # Boost valuable name types
                subtype = artifact.get('subtype', '')
                if subtype == 'username':
                    score += 0.2
                elif subtype == 'pseudonym':
                    score += 0.3
                elif subtype == 'project_name':
                    score += 0.2
                else:
                    score += 0.1  # Generic name boost
                
                # Boost score for artifacts that look like usernames/handles (alphanumeric with underscores)
                if re.match(r'^[a-zA-Z0-9_-]+$', content):
                    score += 0.2
                
                # Further boost GitHub handles and other valuable username formats
                if re.match(r'^[a-zA-Z][a-zA-Z0-9_-]{2,}$', content):
                    score += 0.1
                
                # Penalize names with too many spaces (likely sentence fragments)
                space_ratio = content.count(' ') / len(content) if len(content) > 0 else 0
                score -= space_ratio * 0.4
                
                # Penalize names where a high percentage of words are common stopwords
                if len(words) > 1:
                    stopwords = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
                                'should', 'can', 'could', 'may', 'might', 'must', 'of', 'to', 'from',
                                'by', 'with', 'for', 'in', 'on', 'at', 'as', 'this', 'that', 'these',
                                'those', 'their', 'his', 'her', 'its', 'our', 'your', 'my', 'mine']
                    stopword_count = sum(1 for word in words if word.lower() in stopwords)
                    if stopword_count / len(words) > 0.3:  # More than 30% stopwords
                        score -= 0.3
            else:
                # For non-name artifacts, use standard scoring boosts
                if artifact_type in ['username', 'alias', 'wallet_address', 'private_key']:
                    score += 0.2
            
            # Skip artifacts that are low-scoring after adjustments
            if score < 0.3:  # Increased minimum threshold
                logger.debug(f"Skipping low-scoring artifact: {content} (score: {score})")
                skipped_count += 1
                continue
                
            # Update the artifact score with our adjustments
            artifact['score'] = min(1.0, score)  # Cap at 1.0
            
            # Create a discovery from the artifact
            discovery = {
                'id': artifact.get('hash', str(len(self.discoveries) + len(new_discoveries) + 1)),
                'type': artifact.get('type', 'unknown'),
                'content': content,
                'summary': artifact.get('summary', ''),
                'source_url': source_url,
                'original_url': original_url if is_wayback else source_url,
                'is_wayback': is_wayback,
                'date': artifact.get('date'),
                'score': artifact.get('score', 0),
                'timestamp': self._get_timestamp(),
                'iteration': self.current_iteration,
                'name': name  # Preserve the name field if present
            }
            
            # Check for duplicates against all previous discoveries
            logger.debug(f"Checking duplicate for: {discovery['summary']}")
            if self._is_duplicate_discovery(discovery):
                # Log with more details about what was skipped to help with debugging
                if discovery.get('type') == 'name':
                    logger.debug(f"Skipping duplicate: '{content}' (source: {source_url})")
                else:
                    logger.debug(f"Skipping duplicate: {content[:40]}... (type: {discovery.get('type')})")
                skipped_count += 1
                continue
                
            # If we get here, this is a new, unique discovery
            logger.info(f"Adding new discovery: {content[:40]}... (type: {discovery.get('type')})")
            self.discoveries.append(discovery)
            new_discoveries.append(discovery)
            
            # Update entity aliases if this is a name-related discovery
            if artifact_type in ['username', 'alias', 'wallet_address', 'name']:
                self.entity_aliases.add(content)
            
            # Log the new discovery
            logger.info(f"New unique discovery: {discovery['type']} - {discovery['summary']}")
        
        # Log summary information with more details
        logger.info(f"Processed {len(artifacts)} artifacts from {source_url}")
        logger.info(f"Added {len(new_discoveries)} new discoveries, filtered {len(artifacts) - len(new_discoveries)} duplicates/low-quality")
        
        # Add more detailed summary for name artifacts to help with debugging
        name_artifacts = [d for d in new_discoveries if d.get('type') == 'name']
        if name_artifacts:
            # Group by subtype for better organization
            by_subtype = {}
            for artifact in name_artifacts:
                subtype = artifact.get('subtype', 'unknown')
                if subtype not in by_subtype:
                    by_subtype[subtype] = []
                by_subtype[subtype].append(artifact)
            
            # Log summary by subtype
            logger.info(f"New unique name artifacts ({len(name_artifacts)}) by type:")
            for subtype, artifacts in by_subtype.items():
                logger.info(f"  {subtype} ({len(artifacts)}):")
                # Sort by score (highest first)
                sorted_artifacts = sorted(artifacts, key=lambda x: x.get('score', 0), reverse=True)
                # Show top 5 per subtype
                for artifact in sorted_artifacts[:5]:
                    logger.info(f"    - '{artifact.get('content', '')}' (score: {artifact.get('score', 0)})")
                if len(sorted_artifacts) > 5:
                    logger.info(f"    - ... and {len(sorted_artifacts) - 5} more")
                    
            # Show top names across all subtypes
            logger.info(f"Top 10 highest-scoring name artifacts overall:")
            top_artifacts = sorted(name_artifacts, key=lambda x: x.get('score', 0), reverse=True)[:10]
            for artifact in top_artifacts:
                logger.info(f"  - '{artifact.get('content', '')}' ({artifact.get('subtype', 'unknown')}, score: {artifact.get('score', 0)})")
                
            # Log the duplicate count
            logger.info(f"Skipped {skipped_count} duplicate or filtered artifacts")
        
        return new_discoveries
    
    def _consult_llm_for_next_steps(self, discoveries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Consult the LLM for next investigation steps based on recent discoveries.
        
        Args:
            discoveries: Recent discoveries to analyze
            
        Returns:
            List of new investigation targets
        """
        if not discoveries:
            return []
        
        # Prepare context for the LLM
        context = self._prepare_llm_context(discoveries)
        
        # Build the prompt
        prompt = f"""
You are an expert digital detective investigating: "{self.objective}"

Recent findings:
{context}

Based on these findings, I need:
1. 3-5 specific URLs I should investigate next (with explanation for each)
2. 2-3 search queries I should run (with explanation for each)
3. Any specific archives, forums, or repositories to check
4. Any historical time periods I should focus on for Wayback Machine snapshots

Format your suggestions as a JSON object with the following structure:
{{
    "website_targets": [
        {{"url": "https://example.com/path", "rationale": "Explanation of why this is relevant"}}
    ],
    "search_queries": [
        {{"query": "example search query", "rationale": "Explanation of why this search would be valuable"}}
    ],
    "wayback_targets": [
        {{"url": "https://example.com", "year_range": [2013, 2016], "rationale": "Explanation of why checking these historical snapshots matters"}}
    ],
    "github_targets": [
        {{"url": "https://github.com/username/repo", "rationale": "Explanation of what to look for in this repository"}}
    ]
}}

Be specific, not generic. Use exact URLs and search terms tailored to our objective.
"""
        
        # Call the LLM with alternating models
        try:
            # Try Claude first
            llm = self._get_llm_instance()
            response = None
            
            if llm.use_claude:
                response = llm._call_claude(prompt)
                
            # If Claude failed, try OpenAI
            if not response or response.strip() == "{}":
                logger.warning("Claude failed, trying OpenAI as backup")
                try:
                    backup_llm = LLMIntegration(use_claude=False, use_openai=True)
                    response = backup_llm._call_openai(prompt)
                except Exception as e:
                    logger.warning(f"OpenAI also failed: {str(e)}")
                    response = None
                    
            # If both Claude and OpenAI failed, try Gemini
            if not response or response.strip() == "{}":
                logger.warning("Claude and OpenAI failed, trying Gemini as final backup")
                try:
                    gemini_llm = LLMIntegration(use_claude=False, use_openai=False, use_gemini=True)
                    response = gemini_llm._call_gemini(prompt)
                except Exception as e:
                    logger.error(f"All LLM services failed. Gemini error: {str(e)}")
                    response = "{}"
                
            json_str = llm._extract_json(response)
            suggestions = json.loads(json_str)
            
            # Log the LLM suggestions
            logger.debug(f"LLM suggestions: {json.dumps(suggestions, indent=2)}")
            
            # Convert suggestions to investigation targets
            new_targets = []
            
            # Process website targets
            for target in suggestions.get('website_targets', []):
                url = target.get('url')
                if url and self._is_valid_url(url) and url not in self.investigated_urls:
                    new_targets.append({
                        'url': url,
                        'type': 'website',
                        'priority': 9,
                        'rationale': target.get('rationale', 'LLM suggestion'),
                        'use_wayback': self._should_check_wayback(url)
                    })
            
            # Process search queries
            for query in suggestions.get('search_queries', []):
                query_text = query.get('query')
                if query_text and f"search:{query_text}" not in self.investigated_urls:
                    new_targets.append({
                        'query': query_text,
                        'type': 'search',
                        'priority': 8,
                        'rationale': query.get('rationale', 'LLM suggestion'),
                        'engine': 'google'
                    })
            
            # Process Wayback targets
            for wayback in suggestions.get('wayback_targets', []):
                url = wayback.get('url')
                if url and self._is_valid_url(url) and f"wayback:{url}" not in self.investigated_urls:
                    new_targets.append({
                        'url': url,
                        'type': 'wayback',
                        'priority': 7,
                        'rationale': wayback.get('rationale', 'LLM suggestion for historical analysis'),
                        'year_range': wayback.get('year_range', [2013, datetime.datetime.now().year])
                    })
            
            # Process GitHub targets
            for github in suggestions.get('github_targets', []):
                url = github.get('url')
                if url and self._is_valid_url(url) and url not in self.investigated_urls:
                    new_targets.append({
                        'url': url,
                        'type': 'github',
                        'priority': 8,
                        'rationale': github.get('rationale', 'LLM suggestion for GitHub repository'),
                        'use_wayback': False
                    })
            
            # Log the LLM consultation
            self._log_to_research_log({
                'event': 'llm_consultation',
                'timestamp': self._get_timestamp(),
                'prompt': prompt,
                'response': response,
                'suggestions': suggestions,
                'new_targets': new_targets
            })
            
            return new_targets
            
        except Exception as e:
            logger.error(f"Error consulting LLM for next steps: {str(e)}")
            return []
    
    def _prepare_llm_context(self, discoveries: List[Dict[str, Any]]) -> str:
        """Prepare context for the LLM based on recent discoveries."""
        context_lines = []
        
        for i, discovery in enumerate(discoveries):
            discovery_type = discovery.get('type', 'unknown')
            summary = discovery.get('summary', '')
            source = discovery.get('source_url', '')
            
            context_line = f"{i+1}. [{discovery_type}] {summary} (Source: {source})"
            context_lines.append(context_line)
        
        # Add information about the entity aliases we've found
        if len(self.entity_aliases) > 1:
            aliases_str = ", ".join(f'"{alias}"' for alias in self.entity_aliases if alias != self.entity)
            if aliases_str:
                context_lines.append(f"\nKnown aliases for {self.entity}: {aliases_str}")
        
        return "\n".join(context_lines)
    
    def _generate_new_leads(self):
        """Generate new investigation leads when the queue is empty."""
        logger.info("Generating new investigation leads...")
        
        # Prepare context with all discoveries so far
        context = self._prepare_llm_context(self.discoveries)
        
        # Build the prompt
        prompt = f"""
You are an expert digital detective investigating: "{self.objective}"

So far, we've discovered:
{context}

We've hit a dead end and need fresh ideas. Based on what we've found so far:

1. What alternative sources should we check that we might have missed?
2. What new search strategies could yield more information?
3. Are there any connections between our findings that suggest new avenues to explore?
4. What historical periods or archives might contain relevant information?

Format your suggestions as a JSON object with the following structure:
{{
    "website_targets": [
        {{"url": "https://example.com/path", "rationale": "Explanation of why this is relevant"}}
    ],
    "search_queries": [
        {{"query": "example search query", "rationale": "Explanation of why this search would be valuable"}}
    ],
    "wayback_targets": [
        {{"url": "https://example.com", "year_range": [2013, 2016], "rationale": "Explanation of why checking these historical snapshots matters"}}
    ],
    "github_targets": [
        {{"url": "https://github.com/username/repo", "rationale": "Explanation of what to look for in this repository"}}
    ]
}}

Be creative and specific. Think of sources we haven't tried yet.
"""
        
        # Call the LLM with alternating models
        try:
            # Try Claude first
            llm = self._get_llm_instance()
            response = None
            
            if llm.use_claude:
                response = llm._call_claude(prompt)
                
            # If Claude failed, try OpenAI
            if not response or response.strip() == "{}":
                logger.warning("Claude failed, trying OpenAI as backup")
                try:
                    backup_llm = LLMIntegration(use_claude=False, use_openai=True)
                    response = backup_llm._call_openai(prompt)
                except Exception as e:
                    logger.warning(f"OpenAI also failed: {str(e)}")
                    response = None
                    
            # If both Claude and OpenAI failed, try Gemini
            if not response or response.strip() == "{}":
                logger.warning("Claude and OpenAI failed, trying Gemini as final backup")
                try:
                    gemini_llm = LLMIntegration(use_claude=False, use_openai=False, use_gemini=True)
                    response = gemini_llm._call_gemini(prompt)
                except Exception as e:
                    logger.error(f"All LLM services failed. Gemini error: {str(e)}")
                    return  # If all LLMs fail, we can't generate new leads
                
            json_str = llm._extract_json(response)
            suggestions = json.loads(json_str)
            
            # Convert suggestions to investigation targets
            new_targets = []
            
            # Process website targets
            for target in suggestions.get('website_targets', []):
                url = target.get('url')
                if url and self._is_valid_url(url) and url not in self.investigated_urls:
                    new_targets.append({
                        'url': url,
                        'type': 'website',
                        'priority': 7,  # Lower priority for these fallback targets
                        'rationale': target.get('rationale', 'LLM fallback suggestion'),
                        'use_wayback': self._should_check_wayback(url)
                    })
            
            # Process search queries
            for query in suggestions.get('search_queries', []):
                query_text = query.get('query')
                if query_text and f"search:{query_text}" not in self.investigated_urls:
                    new_targets.append({
                        'query': query_text,
                        'type': 'search',
                        'priority': 6,
                        'rationale': query.get('rationale', 'LLM fallback suggestion'),
                        'engine': 'google'
                    })
            
            # Process Wayback targets
            for wayback in suggestions.get('wayback_targets', []):
                url = wayback.get('url')
                if url and self._is_valid_url(url) and f"wayback:{url}" not in self.investigated_urls:
                    new_targets.append({
                        'url': url,
                        'type': 'wayback',
                        'priority': 5,
                        'rationale': wayback.get('rationale', 'LLM fallback suggestion for historical analysis'),
                        'year_range': wayback.get('year_range', [2013, datetime.datetime.now().year])
                    })
            
            # Process GitHub targets
            for github in suggestions.get('github_targets', []):
                url = github.get('url')
                if url and self._is_valid_url(url) and url not in self.investigated_urls:
                    new_targets.append({
                        'url': url,
                        'type': 'github',
                        'priority': 6,
                        'rationale': github.get('rationale', 'LLM fallback suggestion for GitHub repository'),
                        'use_wayback': False
                    })
            
            # Update the research queue
            self._update_research_queue(new_targets)
            
            # Log the new leads generation
            self._log_to_research_log({
                'event': 'new_leads_generation',
                'timestamp': self._get_timestamp(),
                'prompt': prompt,
                'response': response,
                'suggestions': suggestions,
                'new_targets': new_targets
            })
            
            logger.info(f"Generated {len(new_targets)} new investigation leads")
            
        except Exception as e:
            logger.error(f"Error generating new leads: {str(e)}")
    
    def _update_research_queue(self, new_targets: List[Dict[str, Any]]):
        """
        Update the research strategy with new targets.
        
        Args:
            new_targets: List of new investigation targets
        """
        # Filter out targets that have already been investigated
        filtered_targets = []
        
        # Track target IDs and URLs we've already seen in the queue to prevent duplicates
        queue_target_ids = {target.get('id') for target in self.research_queue if target.get('id')}
        queue_urls = {target.get('url') for target in self.research_queue if target.get('url')}
        
        for target in new_targets:
            # Skip targets that have already been investigated
            if target.get('type') == 'website' and target.get('url') in self.investigated_urls:
                continue
            if target.get('type') == 'wayback' and f"wayback:{target.get('url')}" in self.investigated_urls:
                continue
            if target.get('type') == 'search' and f"search:{target.get('query')}" in self.investigated_urls:
                continue
            if target.get('type') == 'github' and target.get('url') in self.investigated_urls:
                continue
                
            # Check for duplicates in the queue based on ID
            target_id = target.get('id')
            if target_id and target_id in queue_target_ids:
                logger.debug(f"Skipping duplicate target ID: {target_id}")
                continue
                
            # Check for duplicates in the queue based on URL
            url = target.get('url')
            if url:
                normalized_url = self._normalize_url_for_deduplication(url)
                if normalized_url in queue_urls:
                    logger.debug(f"Skipping duplicate URL: {url}")
                    continue
                # Add normalized URL to prevent future duplicates
                queue_urls.add(normalized_url)
                
            # If target doesn't have an ID yet, create one
            if not target_id and target.get('type') == 'website' and url:
                target['id'] = f"website:{self._normalize_url_for_deduplication(url)}"
                
            # If we got here, this is a new target we should investigate
            filtered_targets.append(target)
        
        # Add new targets to both research strategy and legacy queue
        self.research_strategy.add_targets(filtered_targets)
        
        # Legacy support - also add to the old research_queue for compatibility
        self.research_queue.extend(filtered_targets)
        self.research_queue.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        # Log the additions
        logger.info(f"Added {len(filtered_targets)} new targets to the research strategy (filtered out {len(new_targets) - len(filtered_targets)} duplicates)")
        
        # Log strategy status
        strategy_status = self.research_strategy.get_status()
        logger.info(f"Research strategy now contains {strategy_status['todo_count']} targets to investigate")
    
    def _get_next_investigation_target(self) -> Optional[Dict[str, Any]]:
        """
        Get the next investigation target using focused, depth-first exploration.
        
        This method implements proper focus-driven exploration by:
        1. Sorting research_queue by priority (highest first)
        2. Returning the highest priority target that matches the current focus target
        3. If no matching targets, starting a new focus on the highest priority target
        
        Returns:
            The next target to investigate, or None if no targets are available
        """
        if not self.research_queue:
            return None
            
        # Sort the research queue by priority (highest first)
        self.research_queue.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        # If we have a current focus target, prioritize targets from that domain
        if hasattr(self, 'current_focus_target') and self.current_focus_target:
            # Check if the current focus is exhausted
            if self._is_target_exhausted(self.current_focus_target):
                logger.info(f"Current focus target {self.current_focus_target} is exhausted, selecting new focus")
                self.current_focus_target = None
            else:
                # Find targets that match our current focus domain
                for target in self.research_queue:
                    if target.get('type') in ['website', 'github', 'wayback']:
                        domain = self._extract_domain(target.get('url', ''))
                        if domain == self.current_focus_target:
                            # Found a target matching our current focus
                            self.research_queue.remove(target)
                            
                            # Log the target selection with priority information
                            logger.info(f"Selected target (matching focus {self.current_focus_target}): {target.get('type')} - " +
                                       (f"{target.get('url')}" if target.get('type') in ['website', 'wayback', 'github'] else
                                        f"{target.get('query')}" if target.get('type') == 'search' else "Unknown") +
                                       f" (Priority: {target.get('priority', 'unknown')})")
                            
                            # If the target has a rationale, log it
                            if 'rationale' in target:
                                logger.info(f"Investigation rationale: {target['rationale']}")
                                
                            return target
        
        # If we don't have a focus target or couldn't find matching targets, pick the highest priority one
        if self.research_queue:
            target = self.research_queue[0]
            self.research_queue.remove(target)
            
            # If this is a website/github/wayback target, set it as our new focus
            if target.get('type') in ['website', 'github', 'wayback']:
                domain = self._extract_domain(target.get('url', ''))
                if domain:
                    self.current_focus_target = domain
                    logger.info(f"Setting new focus target to: {self.current_focus_target}")
            
            # Log the target selection with priority information
            logger.info(f"Selected target (highest priority): {target.get('type')} - " +
                       (f"{target.get('url')}" if target.get('type') in ['website', 'wayback', 'github'] else
                        f"{target.get('query')}" if target.get('type') == 'search' else "Unknown") +
                       f" (Priority: {target.get('priority', 'unknown')})")
            
            # If the target has a rationale, log it
            if 'rationale' in target:
                logger.info(f"Investigation rationale: {target['rationale']}")
                
            return target
            
        return None
    
    def _should_continue_investigation(self) -> bool:
        """Determine if the investigation should continue."""
        # Safety check: if no research queue and no way to generate more targets, stop
        if not self.research_queue and not hasattr(self, '_initial_strategy_attempted'):
            logger.warning("No research queue and no initial strategy - stopping investigation")
            return False
            
        # Check if we've reached the maximum number of iterations
        if self.current_iteration >= self.max_iterations:
            logger.info(f"Reached maximum iterations ({self.max_iterations})")
            return False
        
        # Removed idle iterations check to allow more thorough investigation
        # This allows the agent to process all targets in the research queue
        
        # Check if we've exceeded the maximum runtime
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.max_time_seconds:
            logger.info(f"Reached maximum runtime ({self.max_time_seconds} seconds)")
            return False
        
        return True
    
    def _normalize_content(self, content: str, content_type: str = None) -> str:
        """
        Normalize content for consistent duplicate detection.
        
        Args:
            content: The content string to normalize
            content_type: The type of content being normalized (e.g., 'name', 'username')
            
        Returns:
            Normalized string for comparison
        """
        if not content:
            return ""
        
        # Convert to lowercase
        normalized = content.lower()
        
        # Remove excess whitespace (including tabs, newlines)
        normalized = " ".join(normalized.split())
        
        # Special handling for name artifacts
        if content_type == 'name':
            # More aggressive normalization for names
            # Remove all special characters and punctuation
            normalized = re.sub(r'[^\w\s]', '', normalized)
            
            # Remove common titles and suffixes
            for title in ['mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'sir', 'lord', 'lady']:
                normalized = re.sub(r'^' + title + r'\s+', '', normalized)
                
            # Remove generational suffixes
            for suffix in ['jr', 'sr', 'ii', 'iii', 'iv', 'v']:
                normalized = re.sub(r'\s+' + suffix + r'$', '', normalized)
        else:
            # Standard normalization for other content types
            # Keep some special characters that might be significant for non-name content
            normalized = re.sub(r'[^\w\s\-\._@]', '', normalized)
        
        return normalized.strip()
    
    def _is_duplicate_discovery(self, discovery: Dict[str, Any]) -> bool:
        """
        Check if a discovery is a duplicate of an existing one.
        
        Args:
            discovery: The discovery to check
            
        Returns:
            True if it is a duplicate, False otherwise
        """
        discovery_type = discovery.get('type', 'unknown')
        
        # Check ID for exact matches (faster check first)
        discovery_id = discovery.get('id')
        if discovery_id:
            for existing in self.discoveries:
                if existing.get('id') == discovery_id:
                    logger.info(f"Duplicate discovery detected (ID match): {discovery_id}")
                    return True
        
        # For all artifacts, normalize and check content
        discovery_content = discovery.get('content', '')
        if discovery_content:
            # Use type-specific normalization
            normalized_content = self._normalize_content(discovery_content, content_type=discovery_type)
            # Fast lookup in the set of normalized contents
            if normalized_content and normalized_content in self.unique_discovery_contents:
                logger.info(f"Duplicate discovery detected (content match): {discovery_content[:50]}... [{discovery_type}]")
                return True
        
        # Handle name artifacts with special treatment
        if discovery_type == 'name':
            # Specifically check the 'name' field for name artifacts
            discovery_name = discovery.get('name', '')
            
            # If name field exists and is different from content
            if discovery_name and discovery_name != discovery_content:
                # Normalize with special name handling
                normalized_name = self._normalize_content(discovery_name, content_type='name')
                
                if normalized_name and normalized_name in self.unique_discovery_contents:
                    logger.info(f"Duplicate discovery detected (name field match): {discovery_name}")
                    return True
            
            # Additional check: even if content doesn't match exactly, check if we have very similar names
            # For names like "Enterprise" that might appear with different capitalization or spacing
            if discovery_content and discovery_type == 'name':
                # Get just the core name parts (more aggressive normalization)
                core_name = re.sub(r'[^\w]', '', discovery_content.lower())
                
                # For titles, get the name without titles first
                for title in ['mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'sir', 'lord', 'lady']:
                    pattern = r'^' + title + r'\s+'
                    if re.match(pattern, discovery_content.lower()):
                        # This is a name with a title, extract just the name part
                        name_without_title = re.sub(pattern, '', discovery_content.lower())
                        core_name_without_title = re.sub(r'[^\w]', '', name_without_title)
                        if core_name_without_title:
                            core_name = core_name_without_title
                
                # Check if we have any existing name that matches this core exactly
                # (We don't do partial name matching as that's too aggressive)
                for existing in self.discoveries:
                    if existing.get('type') == 'name':
                        existing_content = existing.get('content', '')
                        if existing_content:
                            # Apply the same title removal logic
                            existing_core = re.sub(r'[^\w]', '', existing_content.lower())
                            for title in ['mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'sir', 'lord', 'lady']:
                                pattern = r'^' + title + r'\s+'
                                if re.match(pattern, existing_content.lower()):
                                    name_without_title = re.sub(pattern, '', existing_content.lower())
                                    core_without_title = re.sub(r'[^\w]', '', name_without_title)
                                    if core_without_title:
                                        existing_core = core_without_title
                            
                            # Check for exact core name match
                            if core_name and existing_core and core_name == existing_core:
                                logger.info(f"Duplicate name detected (core match): '{discovery_content}' matches '{existing_content}'")
                                return True
        
        # Additional check for name artifacts against excluded names
        if discovery_type == 'name' and discovery_content:
            # Check against excluded names (entity and variations)
            for excluded in self.excluded_names:
                # Skip very short excluded names to avoid false positives
                if len(excluded) < 3:
                    continue
                # Check if discovery content contains or is contained by an excluded name
                if (excluded.lower() in discovery_content.lower() or 
                    discovery_content.lower() in excluded.lower()):
                    logger.info(f"Duplicate/excluded name detected (entity match): '{discovery_content}' matches excluded name '{excluded}'")
                    return True
        
        # If we reach here, it's not a duplicate - add normalized content to our set
        if discovery_content:
            normalized_content = self._normalize_content(discovery_content, content_type=discovery_type)
            if normalized_content:
                self.unique_discovery_contents.add(normalized_content)
                
        # Also add the name for name artifacts
        if discovery_type == 'name':
            discovery_name = discovery.get('name', '')
            if discovery_name and discovery_name != discovery_content:
                normalized_name = self._normalize_content(discovery_name, content_type='name')
                if normalized_name:
                    self.unique_discovery_contents.add(normalized_name)
        
        # Periodically log deduplication statistics (every 100 discoveries)
        if len(self.unique_discovery_contents) % 100 == 0:
            logger.info(f"Deduplication statistics: {len(self.unique_discovery_contents)} unique contents tracked")
        
        return False
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid."""
        if not url or not isinstance(url, str):
            return False
        
        # Remove whitespace
        url = url.strip()
        if not url:
            return False
        
        try:
            result = urlparse(url)
            valid_format = all([result.scheme in ('http', 'https'), result.netloc])
            
            if not valid_format:
                logger.debug(f"Invalid URL format: {url}")
                return False
            
            # No hardcoded domain-specific checks
            
            # Check for generic pages without domain-specific hardcoding
            skip_paths = [
                '/login', '/signup', '/features', '/team', '/enterprise',
                '/pricing', '/about', '/site', '/security', '/codespaces',
                '/topics', '/collections', '/trending', '/copilot'
            ]
            
            for skip_path in skip_paths:
                if result.path.startswith(skip_path):
                    logger.warning(f"Skipping generic page: {url}")
                    return False
            
            # Skip generic marketing/feature pages without specific name checks
            if 'features' in url or 'pricing' in url or 'about-us' in url or 'contact' in url:
                logger.warning(f"Skipping generic marketing page: {url}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating URL {url}: {str(e)}")
            return False
    
    def _extract_domain(self, text: str) -> Optional[str]:
        """Extract a domain name from text."""
        # Try to extract a domain from the text
        domain_match = re.search(r'((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9])', text.lower())
        
        if domain_match:
            return domain_match.group(1)
        
        return None
        
    def _detect_site_type(self, url: str, html_content: str) -> str:
        """
        Detect the type of website based on URL patterns and content.
        
        Args:
            url: The URL of the website
            html_content: The HTML content of the website
            
        Returns:
            Site type ('forum', 'blog', 'documentation', 'personal_site', or 'generic')
        """
        url_lower = url.lower()
        
        # Identify content types based on generic patterns only, without hardcoded domains
        if '/forum' in url_lower or '/thread' in url_lower or '/discussion' in url_lower:
            return 'forum'
        elif 'blog.' in url_lower or '/blog' in url_lower or '/article' in url_lower:
            return 'blog'
        elif 'docs.' in url_lower or 'documentation' in url_lower or '/docs/' in url_lower:
            return 'documentation'
        elif '/personal' in url_lower or '/about' in url_lower:
            return 'personal_site'
        else:
            return 'generic'
            
    def _investigate_forum_deeply(self, url: str, html_content: str) -> List[Dict[str, Any]]:
        """
        Investigate forum-type sites deeply by following thread links.
        
        Args:
            url: The URL of the forum
            html_content: The HTML content of the forum
            
        Returns:
            List of discoveries from deep investigation
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        discoveries = []
        
        # Generic forum investigation pattern
        # Try to identify forum thread patterns generically
        logger.info(f"Deep investigation of forum: {url}")
        # Extract thread URLs and user profile links using multiple patterns
        thread_links = soup.find_all('a', href=re.compile(r'topic=\d+|thread|discussion|post'))
        
        # Read first 3 threads for content
        thread_urls = []
        for link in thread_links[:5]:  # Get 5 links, but limit to 3 unique ones
            thread_url = urljoin(url, link['href'])
            # Normalize URL to prevent duplicates
            normalized_url = self._normalize_url_for_deduplication(thread_url)
            if normalized_url not in thread_urls and normalized_url not in self.investigated_urls:
                thread_urls.append(normalized_url)
                
        # Investigate up to 3 unique threads
        for thread_url in thread_urls[:3]:
            logger.info(f"Investigating forum thread: {thread_url}")
            thread_html, _ = fetch_page(thread_url)
            if thread_html:
                thread_artifacts = artifact_detector.extract_artifacts(thread_html, thread_url, objective=self.objective, entity=self.entity)
                thread_discoveries = self._process_artifacts(thread_artifacts, thread_url)
                discoveries.extend(thread_discoveries)
                self.investigated_urls.add(thread_url)
        
        # Generic approach for posts and comments
        logger.info(f"Looking for post and comment links on: {url}")
        post_links = soup.find_all('a', href=re.compile(r'/comments/|/post/|/thread/|/discussion/'))
        
        # Collect unique post URLs
        post_urls = []
        for link in post_links[:5]:  # Get 5 links, but limit to 3 unique ones
            post_url = urljoin(url, link['href'])
            # Normalize URL to prevent duplicates
            normalized_url = self._normalize_url_for_deduplication(post_url)
            if normalized_url not in post_urls and normalized_url not in self.investigated_urls:
                post_urls.append(normalized_url)
        
        # Read first 3 unique posts
        for post_url in post_urls[:3]:
            logger.info(f"Investigating post: {post_url}")
            post_html, _ = fetch_page(post_url)
            if post_html:
                post_artifacts = artifact_detector.extract_artifacts(post_html, post_url, objective=self.objective, entity=self.entity)
                post_discoveries = self._process_artifacts(post_artifacts, post_url)
                discoveries.extend(post_discoveries)
                self.investigated_urls.add(post_url)
        
        return discoveries
        
    def _investigate_blog_deeply(self, url: str, html_content: str) -> List[Dict[str, Any]]:
        """
        Investigate blog-type sites deeply by following article links.
        
        Args:
            url: The URL of the blog
            html_content: The HTML content of the blog
            
        Returns:
            List of discoveries from deep investigation
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        discoveries = []
        
        logger.info(f"Deep investigation of blog site: {url}")
        
        # Find article/post links
        post_links = soup.find_all('a', href=True)
        article_urls = []
        
        for link in post_links:
            href = link['href']
            full_url = urljoin(url, href)
            
            # Look for blog post patterns
            if any(pattern in href for pattern in ['/post/', '/article/', '/blog/', '20']):
                if self._extract_domain(full_url) == self._extract_domain(url):
                    # Normalize URL to prevent duplicates
                    normalized_url = self._normalize_url_for_deduplication(full_url)
                    article_urls.append(normalized_url)
        
        # Remove duplicates while preserving order
        article_urls = list(dict.fromkeys(article_urls))
        logger.info(f"Found {len(article_urls)} potential blog post links on {url}")
        
        # Read first 5 blog posts
        for article_url in article_urls[:5]:
            if article_url not in self.investigated_urls:
                logger.info(f"Investigating blog article: {article_url}")
                article_html, _ = fetch_page(article_url)
                if article_html:
                    article_artifacts = artifact_detector.extract_artifacts(article_html, article_url, objective=self.objective, entity=self.entity)
                    article_discoveries = self._process_artifacts(article_artifacts, article_url)
                    discoveries.extend(article_discoveries)
                    self.investigated_urls.add(article_url)
        
        return discoveries
        
    def _extract_same_domain_links(self, html_content: str, base_url: str) -> List[str]:
        """
        Extract links from the same domain with efficient deduplication.
        
        Args:
            html_content: HTML content to extract links from
            base_url: Base URL for resolving relative links
            
        Returns:
            List of unique normalized URLs from the same domain (limited to 5)
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()  # Use set instead of list for efficient deduplication
        base_domain = self._extract_domain(base_url)
        
        if not base_domain:
            logger.warning(f"Could not extract domain from {base_url}")
            return []
            
        logger.info(f"Extracting same-domain links from {base_url} (domain: {base_domain})")
        
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            full_url = urljoin(base_url, href)
            
            # Skip invalid URLs
            if not self._is_valid_url(full_url):
                continue
                
            # Normalize URL to prevent duplicates
            normalized_url = self._normalize_url_for_deduplication(full_url)
            
            # Check if same domain and not already investigated
            if (self._extract_domain(normalized_url) == base_domain and 
                normalized_url not in self.investigated_urls):
                links.add(normalized_url)
        
        # Convert set to list and limit to 5 links max
        result = list(links)[:5]
        logger.info(f"Found {len(links)} unique same-domain links, returning {len(result)} for investigation")
        return result
        
    def _normalize_url_for_deduplication(self, url: str) -> str:
        """
        Normalize URL to prevent duplicate URLs with different formats.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        from urllib.parse import urlparse, urlunparse
        
        try:
            parsed = urlparse(url)
            
            # Remove fragments (#section) and query params (?sort=hot)
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path.rstrip('/'),  # Remove trailing slashes
                '',  # Remove params
                '',  # Remove query
                ''   # Remove fragment
            ))
            
            return normalized.lower()
        except Exception as e:
            logger.warning(f"Error normalizing URL {url}: {str(e)}")
            return url.lower()
    
    def _should_check_wayback(self, url: str) -> bool:
        """Determine if we should check the Wayback Machine for a URL."""
        # Skip URLs that are already Wayback Machine URLs
        if 'web.archive.org' in url:
            return False
            
        # Always check wayback for domains in our priority list
        domain = urlparse(url).netloc
        
        if domain in self.priority_domains:
            return True
        
        # Only use priority_domains from LLM strategy, no hardcoded domains
        
        # No selectively checking other domains - rely solely on LLM strategy
        interesting_domains = {}
        
        for interesting_domain, valuable_paths in interesting_domains.items():
            if interesting_domain in domain:
                path = urlparse(url).path
                # Only check wayback for specific valuable paths to avoid crawling generic pages
                for valuable_path in valuable_paths:
                    if valuable_path in path:
                        return True
                return False  # Skip other paths for these domains
        
        # No hardcoded name/project checks - rely entirely on LLM strategy
        
        # By default, don't check wayback to avoid too many requests
        return False
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp as a string."""
        return datetime.datetime.now().isoformat()
    
    def _log_investigation_results(self, target: Dict[str, Any], discoveries: List[Dict[str, Any]], 
                                  new_targets: List[Dict[str, Any]]):
        """Log the results of an investigation."""
        log_entry = {
            'event': 'investigation',
            'timestamp': self._get_timestamp(),
            'iteration': self.current_iteration,
            'target': target,
            'discoveries': [
                {
                    'id': d.get('id'),
                    'type': d.get('type'),
                    'summary': d.get('summary'),
                    'source_url': d.get('source_url')
                } for d in discoveries
            ],
            'new_targets': [
                {
                    'type': t.get('type'),
                    'url': t.get('url') if t.get('type') in ['website', 'wayback', 'github'] else None,
                    'query': t.get('query') if t.get('type') == 'search' else None,
                    'rationale': t.get('rationale')
                } for t in new_targets
            ]
        }
        
        self._log_to_research_log(log_entry)
    
    def _log_to_research_log(self, entry: Dict[str, Any]):
        """Log an entry to the research log file."""
        try:
            with open(self.research_log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Error writing to research log: {str(e)}")
    
    def _save_state(self):
        """Save the current state of the investigation."""
        state = {
            'objective': self.objective,
            'entity': self.entity,
            'current_iteration': self.current_iteration,
            'idle_iterations': self.idle_iterations,
            'start_time': self.start_time,
            'discoveries': self.discoveries,
            'entity_aliases': list(self.entity_aliases),
            'priority_domains': list(self.priority_domains),
            'research_queue': self.research_queue,
            'investigated_urls': list(self.investigated_urls),
            'timestamp': self._get_timestamp()
        }
        
        try:
            state_path = os.path.join(self.results_dir, f'investigation_state_{int(time.time())}.json')
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Also save discoveries separately
            discoveries_path = os.path.join(self.results_dir, 'discoveries.json')
            with open(discoveries_path, 'w') as f:
                json.dump(self.discoveries, f, indent=2)
                
            logger.info(f"Saved investigation state to {state_path}")
        except Exception as e:
            logger.error(f"Error saving investigation state: {str(e)}")
    
    def _extract_urls_from_text(self, text: str) -> List[str]:
        """
        Extract URLs from text content.
        
        Args:
            text: The text to extract URLs from
            
        Returns:
            List of extracted URLs
        """
        if not text:
            return []
            
        # Simple regex for URL extraction
        url_pattern = r'https?://[^\s)"\']+'
        urls = re.findall(url_pattern, text)
        
        # Deduplicate URLs
        unique_urls = list(set(urls))
        
        return unique_urls
    
    def _extract_github_urls(self, text: str) -> List[str]:
        """
        Extract GitHub URLs from text content.
        
        Args:
            text: The text to extract GitHub URLs from
            
        Returns:
            List of extracted GitHub URLs
        """
        if not text:
            return []
            
        # Regex specifically for GitHub URLs
        github_pattern = r'https?://(?:www\.)?github\.com/[^\s)"\']+'
        github_urls = re.findall(github_pattern, text)
        
        # Deduplicate URLs
        unique_urls = list(set(github_urls))
        
        return unique_urls
    
    def _is_url_relevant(self, url: str) -> bool:
        """
        Determine if a URL is relevant to the investigation.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL is relevant, False otherwise
        """
        if not url:
            return False
            
        # No hardcoded irrelevant domains - rely on the LLM strategy instead
                
        # Check if URL is in a priority domain
        for domain in self.priority_domains:
            if domain in url.lower():
                return True
                
        # Default to considering it relevant
        return True
    
    def _extract_leads_from_discovery(self, discovery: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract potential investigation leads from a discovery.
        
        Args:
            discovery: A discovery that might contain leads
            
        Returns:
            List of potential investigation leads
        """
        leads = []
        
        # Extract URLs from the discovery content
        if 'content' in discovery:
            content = discovery['content']
            urls = self._extract_urls_from_text(content)
            
            for url in urls:
                # Skip already investigated URLs
                if url in self.investigated_urls:
                    continue
                    
                # Skip URLs that are clearly not relevant
                if not self._is_url_relevant(url):
                    continue
                
                # Add as a potential lead with medium priority
                leads.append({
                    'url': url,
                    'type': 'website',
                    'priority': 6,  # Medium priority
                    'rationale': f'URL found in discovery: {discovery.get("title", "Unknown")}',
                    'discovered_in': discovery.get('id', '')
                })
                
        # Extract potential search queries from discovery
        if 'title' in discovery and discovery.get('artifact_type') == 'name':
            # If we found a name artifact, it might be worth investigating
            name = discovery.get('artifact_value', '')
            if name and len(name) > 3:
                leads.append({
                    'query': f'{name} {self.entity}',
                    'type': 'search',
                    'priority': 7,  # Medium-high priority
                    'rationale': f'Name artifact found: {name}',
                    'discovered_in': discovery.get('id', '')
                })
        
        # Extract GitHub URLs specifically
        if 'content' in discovery:
            github_urls = self._extract_github_urls(discovery['content'])
            for url in github_urls:
                if url in self.investigated_urls:
                    continue
                    
                leads.append({
                    'url': url,
                    'type': 'github',
                    'priority': 8,  # High priority
                    'rationale': f'GitHub URL found in discovery: {discovery.get("title", "Unknown")}',
                    'discovered_in': discovery.get('id', '')
                })
                
        # Extract wayback investigation opportunities
        if 'url' in discovery and discovery.get('source_type') == 'website':
            # Add a wayback investigation lead for this URL
            leads.append({
                'url': discovery['url'],
                'type': 'wayback',
                'priority': 7,
                'rationale': f'Historical investigation of discovered URL: {discovery["url"]}',
                'year_range': (2013, datetime.datetime.now().year)
            })
            
        logger.info(f"Extracted {len(leads)} potential leads from discovery: {discovery.get('title', 'Unknown')}")
        return leads
    
    def _review_research_strategy(self, force_update: bool = False):
        """
        Perform a comprehensive review of the research strategy.
        
        Args:
            force_update: If True, force a strategy update even if not scheduled
        """
        # Skip if not forced and we've recently updated
        if not force_update and time.time() - self.research_strategy.last_update_time < 600:  # 10 minutes
            return
            
        logger.info("Performing comprehensive research strategy review...")
        
        # Get current strategy status
        strategy_status = self.research_strategy.get_status()
        
        # Prepare discoveries context
        recent_discoveries = []
        for iteration in range(max(0, self.current_iteration - 10), self.current_iteration + 1):
            if iteration in self.iteration_discoveries:
                recent_discoveries.extend(self.iteration_discoveries[iteration])
                
        # Limit to most recent discoveries
        recent_discoveries = recent_discoveries[-10:]
        
        # Build context for the LLM
        context = {
            'objective': self.objective,
            'entity': self.entity,
            'current_iteration': self.current_iteration,
            'total_discoveries': len(self.discoveries),
            'recent_discoveries': recent_discoveries,
            'current_strategy': strategy_status,
            'completed_targets': self.research_strategy.completed_targets[-10:],  # Last 10 completed targets
            'todo_count': strategy_status['todo_count']
        }
        
        # Get strategy recommendations from LLM
        strategy_updates = self._get_strategy_recommendations(context)
        
        # Apply the updates to our strategy
        if strategy_updates:
            self.research_strategy.merge_strategy_updates(strategy_updates)
            
        logger.info(f"Research strategy updated. New todo count: {self.research_strategy.get_status()['todo_count']}")
    
    def _get_strategy_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get strategy recommendations from the LLM.
        
        Args:
            context: Context information for the LLM
            
        Returns:
            Dictionary with strategy updates
        """
        # Use alternating LLM for this call
        llm = self._get_llm_instance()
        
        # Build the prompt
        prompt = f"""
You are an expert research strategist helping investigate: "{self.objective}"

CURRENT INVESTIGATION STATUS:
- Iteration: {context['current_iteration']}
- Total discoveries: {context['total_discoveries']}
- Remaining todo targets: {context['todo_count']}
- Target types investigated: {', '.join(context.get('current_strategy', {}).get('target_types', []))}

RECENT DISCOVERIES:
{self._format_discoveries_for_prompt(context.get('recent_discoveries', []))}

RECENTLY COMPLETED TARGETS:
{self._format_targets_for_prompt(context.get('completed_targets', []))}

I need you to perform a comprehensive review of my research strategy. Your job is to:

1. Evaluate the productivity of different target types
2. Suggest specific new high-priority targets that I should investigate
3. Recommend priority adjustments for different target types
4. Identify gaps in the investigation strategy

Please provide a focused, comprehensive research plan that prioritizes depth-first exploration.
Return ONLY a JSON object with your recommendations in this format:

```json
{{
  "strategy_analysis": "Brief analysis of current strategy strengths/weaknesses",
  "recommended_approach": "Specific recommendation for how to proceed",
  "new_targets": [
    {{
      "type": "website",
      "url": "https://specific-example.com/exact-page",
      "priority": 9,
      "rationale": "Specific reason this target is valuable"
    }},
    ...more targets...
  ],
  "priority_updates": {{
    "type_adjustments": {{
      "github": 2,
      "wayback": 1,
      "website": 0,
      "search": -1
    }},
    "domain_priorities": {{
      "example-high-priority.com": 10,
      "example-medium-priority.org": 7
    }}
  }}
}}
```

Focus on identifying the MOST valuable unexplored targets and making the strategy more methodical.
"""
        
        # Get the response
        if llm.use_claude:
            response = llm._call_claude(prompt)
        elif llm.use_openai:
            response = llm._call_openai(prompt)
        else:
            logger.error("No LLM service available")
            return {}
            
        # Extract JSON from the response
        try:
            json_str = llm._extract_json(response)
            recommendations = json.loads(json_str)
            
            # Log key recommendations
            logger.info(f"Strategy analysis: {recommendations.get('strategy_analysis', 'No analysis provided')}")
            logger.info(f"Recommended approach: {recommendations.get('recommended_approach', 'No approach provided')}")
            logger.info(f"New targets suggested: {len(recommendations.get('new_targets', []))}")
            
            return recommendations
        except Exception as e:
            logger.error(f"Error parsing LLM strategy recommendations: {e}")
            logger.debug(f"Raw LLM response: {response}")
            return {}
    
    def _format_discoveries_for_prompt(self, discoveries: List[Dict[str, Any]]) -> str:
        """Format discoveries for inclusion in a prompt."""
        if not discoveries:
            return "No recent discoveries."
            
        result = []
        for i, discovery in enumerate(discoveries[:5]):  # Limit to 5 discoveries
            discovery_type = discovery.get('artifact_type', 'unknown')
            title = discovery.get('title', 'Untitled')
            source = discovery.get('source', 'Unknown source')
            
            result.append(f"{i+1}. [{discovery_type}] {title} (from {source})")
            
        if len(discoveries) > 5:
            result.append(f"...and {len(discoveries) - 5} more discoveries")
            
        return "\n".join(result)
    
    def _format_targets_for_prompt(self, targets: List[Dict[str, Any]]) -> str:
        """Format targets for inclusion in a prompt."""
        if not targets:
            return "No recently completed targets."
            
        result = []
        for i, target in enumerate(targets[:5]):  # Limit to 5 targets
            target_type = target.get('type', 'unknown')
            
            if target_type == 'website' and 'url' in target:
                info = target['url']
            elif target_type == 'search' and 'query' in target:
                info = target['query']
            elif target_type == 'wayback' and 'url' in target:
                info = f"Wayback: {target['url']}"
            elif target_type == 'github' and 'url' in target:
                info = target['url']
            else:
                info = "Unknown target"
                
            discoveries = target.get('discoveries_count', 0)
            result.append(f"{i+1}. [{target_type}] {info} (Discoveries: {discoveries})")
            
        if len(targets) > 5:
            result.append(f"...and {len(targets) - 5} more targets")
            
        return "\n".join(result)
    
    def _generate_investigation_report(self):
        """Generate a final report of the investigation."""
        report = {
            'objective': self.objective,
            'entity': self.entity,
            'iterations': self.current_iteration,
            'total_discoveries': len(self.discoveries),
            'entity_aliases': list(self.entity_aliases),
            'start_time': datetime.datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': self._get_timestamp(),
            'runtime_seconds': time.time() - self.start_time,
            'discoveries_by_type': {},
            'top_discoveries': [],
            'summary': ""
        }
        
        # Count discoveries by type
        for discovery in self.discoveries:
            discovery_type = discovery.get('type', 'unknown')
            if discovery_type not in report['discoveries_by_type']:
                report['discoveries_by_type'][discovery_type] = 0
            report['discoveries_by_type'][discovery_type] += 1
        
        # Get top discoveries (highest scoring)
        sorted_discoveries = sorted(self.discoveries, key=lambda x: x.get('score', 0), reverse=True)
        report['top_discoveries'] = [
            {
                'type': d.get('type'),
                'summary': d.get('summary'),
                'source_url': d.get('source_url'),
                'score': d.get('score', 0)
            } for d in sorted_discoveries[:10]  # Top 10 discoveries
        ]
        
        # Generate an investigation summary using the LLM
        report['summary'] = self._generate_investigation_summary()
        
        # Save the report
        try:
            report_path = os.path.join(self.results_dir, f'investigation_report_{int(time.time())}.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Saved investigation report to {report_path}")
        except Exception as e:
            logger.error(f"Error saving investigation report: {str(e)}")
    
    def _generate_investigation_summary(self) -> str:
        """Generate a summary of the investigation using the LLM."""
        # Prepare context with top discoveries
        sorted_discoveries = sorted(self.discoveries, key=lambda x: x.get('score', 0), reverse=True)
        top_discoveries = sorted_discoveries[:20]  # Use top 20 for the summary
        
        context = self._prepare_llm_context(top_discoveries)
        
        # Build the prompt
        prompt = f"""
You are an expert digital detective who has completed an investigation on: "{self.objective}"

Here are the most significant findings from the investigation:
{context}

Please provide a detailed summary of the investigation results, including:
1. A high-level overview of what was discovered
2. The most important findings and their significance
3. Connections between different discoveries
4. Conclusions that can be drawn from the evidence

Write in a professional, analytical tone appropriate for an investigation report.
"""
        
        # Call the LLM with alternating models
        try:
            # Try Claude first
            llm = self._get_llm_instance()
            summary = None
            
            if llm.use_claude:
                summary = llm._call_claude(prompt)
                
            # If Claude failed, try OpenAI
            if not summary or summary.strip() == "":
                logger.warning("Claude failed, trying OpenAI as backup")
                try:
                    backup_llm = LLMIntegration(use_claude=False, use_openai=True)
                    summary = backup_llm._call_openai(prompt)
                except Exception as e:
                    logger.warning(f"OpenAI also failed: {str(e)}")
                    summary = None
                    
            # If both Claude and OpenAI failed, try Gemini
            if not summary or summary.strip() == "":
                logger.warning("Claude and OpenAI failed, trying Gemini as final backup")
                try:
                    gemini_llm = LLMIntegration(use_claude=False, use_openai=False, use_gemini=True)
                    summary = gemini_llm._call_gemini(prompt)
                except Exception as e:
                    logger.error(f"All LLM services failed. Gemini error: {str(e)}")
                    summary = "Failed to generate investigation summary due to LLM service errors."
                
            return summary
        except Exception as e:
            logger.error(f"Error generating investigation summary: {str(e)}")
            return "Failed to generate investigation summary."


def main():
    """Main function to run the detective agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Detective Agent")
    parser.add_argument("--objective", type=str, default="Find name artifacts around Vitalik Buterin",
                        help="The research objective")
    parser.add_argument("--entity", type=str, default="Vitalik Buterin",
                        help="The primary entity to investigate")
    parser.add_argument("--max-iterations", type=int, default=50,
                        help="Maximum number of investigation iterations")
    parser.add_argument("--max-time-hours", type=float, default=24.0,
                        help="Maximum runtime in hours")
    parser.add_argument("--max-idle-iterations", type=int, default=5,
                        help="Maximum number of iterations without new discoveries")
    
    args = parser.parse_args()
    
    # Create and run the detective agent
    detective = DetectiveAgent(
        objective=args.objective,
        entity=args.entity,
        max_iterations=args.max_iterations,
        max_time_hours=args.max_time_hours,
        max_idle_iterations=args.max_idle_iterations
    )
    
    discoveries = detective.start_investigation()
    
    logger.info(f"Investigation complete. Found {len(discoveries)} discoveries.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())