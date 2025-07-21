#!/usr/bin/env python3
"""
Detective Agent - Main entry point for the Narrahunt investigation system

This slim version of the detective agent imports functionality from the modules directory.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core functionality from modules
from modules.agent_core import main as run_detective
from modules.agent_core import start_investigation, execute_investigation
from modules.routing import review_research_strategy, extract_leads_from_discovery
from modules.extractors import extract_artifacts_from_html, extract_names_from_text
from modules.llm_engine import get_llm_instance, alternate_llm_instance

class DetectiveAgent:
    """
    Main detective agent class for Narrahunt Phase 2.
    
    This class serves as a unified interface to the modularized detective functionality.
    """
    
    def __init__(self, objective: str, entity: str, max_iterations: int = 50,
                 max_idle_iterations: int = 10, save_path: Optional[str] = None):
        """
        Initialize the detective agent.
        
        Args:
            objective: The research objective
            entity: The primary entity to investigate
            max_iterations: Maximum number of investigation iterations
            max_idle_iterations: Maximum consecutive iterations without new discoveries
            save_path: Path to save discoveries and state
        """
        self.objective = objective
        self.entity = entity
        self.max_iterations = max_iterations
        self.max_idle_iterations = max_idle_iterations
        self.save_path = save_path or os.path.join(os.getcwd(), "discoveries")
        
        # Ensure save directory exists
        os.makedirs(self.save_path, exist_ok=True)
        
        # Initialize state
        self.current_iteration = 0
        self.idle_iterations = 0
        self.discoveries = []
        self.iteration_discoveries = {}
        self.current_target = None
        self.llm_calls_count = 0
        
        # Initialize research strategy
        # Note: In the full version, this would initialize a ResearchStrategy object
        self.research_strategy = None
        
        logger.info(f"Detective Agent initialized with objective: {objective}")
        logger.info(f"Primary entity: {entity}")
        
    def start_investigation(self):
        """Start the investigation process."""
        # Call the modularized start_investigation function
        return start_investigation(self)
    
    def _get_llm_instance(self):
        """Get an LLM instance, alternating between providers."""
        return alternate_llm_instance(self.llm_calls_count)

# If run directly, execute the detective agent
if __name__ == "__main__":
    sys.exit(run_detective())