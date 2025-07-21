#!/usr/bin/env python3
"""
Integration script to connect the Narrative Discovery Matrix with the existing crawler system.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add the project directory to the path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Import the Narrative Matrix and Objectives Manager
from narrative_matrix import NarrativeMatrix
from objectives_manager import ObjectivesManager

# Import existing crawler components
try:
    from url_queue import URLQueue
    from crawl import extract_links
    from fetch import fetch_url
    from artifact_extractor import extract_artifacts_from_html
    
    crawl_components_available = True
except ImportError as e:
    crawl_components_available = False
    print(f"Warning: Could not import crawler components: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_dir, 'logs', 'integration.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('narrahunt.integration')

class IntegratedSystem:
    """
    Integrates the Narrative Discovery Matrix with the existing crawler system.
    """
    
    def __init__(self):
        """Initialize the integrated system."""
        # Initialize matrix and objectives manager
        self.matrix = NarrativeMatrix()
        self.manager = ObjectivesManager()
        
        # Initialize crawler components if available
        self.url_queue = None
        if crawl_components_available:
            try:
                self.url_queue = URLQueue()
                logger.info("Successfully initialized crawler components")
            except Exception as e:
                logger.error(f"Error initializing crawler components: {e}")
    
    def run_integrated_cycle(self, max_cycles=5, cycle_delay=300):
        """
        Run an integrated cycle of matrix-driven objective generation and crawling.
        
        Args:
            max_cycles: Maximum number of objective cycles to run
            cycle_delay: Delay in seconds between cycles
        """
        logger.info(f"Starting integrated cycle with max_cycles={max_cycles}")
        
        for cycle in range(max_cycles):
            logger.info(f"Starting cycle {cycle+1}/{max_cycles}")
            
            # Get the current objective or generate a new one
            objective = self.manager.load_current_objective()
            if not objective:
                logger.error("Failed to load or generate an objective. Stopping cycle.")
                break
            
            logger.info(f"Working on objective: {objective}")
            
            # Parse the objective to extract artifact_type and entity
            words = objective.split()
            artifact_type = None
            entity = None
            
            for i, word in enumerate(words):
                if word.lower() in ["find", "discover"] and i+1 < len(words):
                    artifact_type = words[i+1]
                if word.lower() in ["around", "related", "associated", "connected"] and i+1 < len(words):
                    entity = " ".join(words[i+1:])
                    # Remove any trailing punctuation
                    entity = entity.rstrip(".,:;")
            
            logger.info(f"Parsed objective - Artifact type: {artifact_type}, Entity: {entity}")
            
            # Use the entity and artifact type as search seeds
            if crawl_components_available and self.url_queue:
                # Clear the existing queue
                self.url_queue.clear()
                
                # Add matrix-generated search terms
                search_terms = []
                if entity:
                    search_terms.append(entity)
                if artifact_type:
                    search_terms.append(artifact_type)
                if entity and artifact_type:
                    search_terms.append(f"{entity} {artifact_type}")
                
                for term in search_terms:
                    # Add DuckDuckGo search
                    self.url_queue.add(f"https://duckduckgo.com/html/?q={term.replace(' ', '+')}")
                    
                    # Add GitHub search
                    self.url_queue.add(f"https://github.com/search?q={term.replace(' ', '+')}&type=code")
                    
                    # Add other search engines or specific sites as needed
                
                logger.info(f"Added {len(search_terms)} matrix-generated search terms to the queue")
                
                # Process the queue
                processed_count = 0
                discoveries = []
                
                while not self.url_queue.is_empty() and processed_count < 50:  # Limit to 50 URLs per cycle
                    url = self.url_queue.next()
                    if not url:
                        break
                    
                    logger.info(f"Processing URL: {url}")
                    try:
                        # Fetch the page content
                        content, mime_type = fetch_url(url)
                        if content and mime_type and 'html' in mime_type.lower():
                            # Extract artifacts
                            artifacts = extract_artifacts_from_html(content, url, date=datetime.now().isoformat())
                            
                            # Record discoveries
                            for artifact in artifacts:
                                discovery = {
                                    "source": "crawler",
                                    "url": url,
                                    "content": artifact.get("summary", ""),
                                    "artifact_type": artifact.get("type", "unknown"),
                                    "score": artifact.get("score", 0),
                                    "entities": [entity] if entity else [],
                                    "related_artifacts": [artifact_type] if artifact_type else []
                                }
                                
                                # Record the discovery in the matrix
                                # Determine if it's narrative-worthy based on score
                                narrative_worthy = artifact.get("score", 0) > 0.7
                                self.matrix.record_discovery(discovery, narrative_worthy=narrative_worthy)
                                discoveries.append(discovery)
                            
                            # Extract and queue links
                            links = extract_links(url, content)
                            for link in links:
                                self.url_queue.add(link)
                        
                        processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing URL {url}: {e}")
                
                logger.info(f"Processed {processed_count} URLs, found {len(discoveries)} discoveries")
                
                # Check if we should move to the next objective
                if self.manager.is_objective_exhausted():
                    logger.info("Objective is exhausted, generating related objectives")
                    
                    # Generate related objectives
                    related_objectives = self.manager.generate_related_objectives()
                    for related in related_objectives:
                        logger.info(f"Related objective: {related}")
                    
                    # Move to the next objective
                    next_objective = self.manager.move_to_next_objective()
                    logger.info(f"Moved to next objective: {next_objective}")
            else:
                logger.warning("Crawler components not available. Skipping crawl phase.")
                
                # If we can't crawl, simulate some progress for testing
                # In a real system, you'd want to skip to the next objective
                time.sleep(5)
                
                # Move to the next objective
                next_objective = self.manager.move_to_next_objective()
                logger.info(f"Moved to next objective: {next_objective}")
            
            # Wait before the next cycle
            if cycle < max_cycles - 1:
                logger.info(f"Waiting {cycle_delay} seconds before next cycle")
                time.sleep(cycle_delay)
        
        logger.info("Completed integrated cycle")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'), exist_ok=True)
    
    # Initialize and run the integrated system
    system = IntegratedSystem()
    
    # Display system status
    print("Narrative Discovery Matrix - Integrated System")
    print("=============================================")
    print(f"Crawler components available: {crawl_components_available}")
    
    # Run a short integrated cycle as a test
    system.run_integrated_cycle(max_cycles=2, cycle_delay=30)
    
    print("System is ready for integrated operation")