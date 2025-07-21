#!/usr/bin/env python3
"""
Integrated Main Controller for Narrahunt Phase 2.

This controller integrates the Narrative Discovery Matrix with the existing
crawler infrastructure, using matrix-generated objectives to guide research.
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# Set up base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Set up logging
os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_dir, 'logs', 'integrated_main.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('narrahunt.integrated_main')

# Import core matrix components
from core.narrative_matrix import NarrativeMatrix
from core.objectives_manager import ObjectivesManager

# Import crawler components
from core.url_queue import URLQueue
from core.fetch import fetch_page
from core.crawl import extract_links, is_allowed_by_robots

# Import enhanced components
from core.llm_research_strategy import LLMResearchStrategy
from core.wayback_integration import WaybackMachine
from core.enhanced_artifact_detector import EnhancedArtifactDetector

class IntegratedController:
    """
    Integrated controller that connects the Narrative Discovery Matrix
    with the crawler infrastructure.
    """
    
    def __init__(self):
        """Initialize the integrated controller."""
        # Initialize core components
        self.matrix = NarrativeMatrix()
        self.objectives_manager = ObjectivesManager()
        
        # Initialize crawler components with a local state file
        state_file = os.path.join(base_dir, 'cache', 'url_queue_state.json')
        self.url_queue = URLQueue(state_file=state_file)
        
        # Initialize enhanced components
        self.llm_strategy = LLMResearchStrategy()
        self.wayback_machine = WaybackMachine()
        self.artifact_detector = EnhancedArtifactDetector()
        
        # Load configuration
        self.config = self.load_config()
        
        # Track statistics
        self.stats = {
            "objectives_processed": 0,
            "urls_processed": 0,
            "artifacts_found": 0,
            "high_scoring_artifacts": 0,
            "narrative_worthy_discoveries": 0,
            "start_time": None,
            "end_time": None,
            "sources_accessed": Counter(),
            "wayback_snapshots_accessed": 0
        }
        
        # Path for discovery log
        self.discovery_log_path = os.path.join(base_dir, 'results', 'discovery_log.txt')
        self.session_summaries_dir = os.path.join(base_dir, 'results', 'session_summaries')
    
    def load_config(self, config_path=os.path.join(base_dir, "config", "integrated_config.json")) -> Dict[str, Any]:
        """
        Load configuration from file or create default.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Default configuration
        default_config = {
            "crawl_delay": 2,
            "max_pages_per_objective": 100,
            "max_depth": 3,
            "respect_robots": True,
            "follow_redirects": True,
            "wayback_integration": True,
            "llm_research_strategy": True,
            "enhanced_artifact_detection": True,
            "allowed_domains": [
                # Removed hardcoded fallback domains - use LLM strategy only
                "web.archive.org",
                "medium.com",
                "twitter.com",
                "reddit.com",
                "duckduckgo.com"
            ]
        }
        
        # Save default configuration
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def generate_discovery_summary(self, objective: str, results: Dict[str, Any]) -> str:
        """
        Generate a summary of discoveries for an objective.
        
        Args:
            objective: The research objective
            results: Results dictionary from process_objective
            
        Returns:
            Summary text string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        artifact_type, entity = self.parse_objective(objective)
        
        # Count domain sources
        domain_counts = dict(self.stats.get("sources_accessed", Counter()))
        top_sources = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5] if domain_counts else []
        
        # Get narratives directory and count narrative-worthy discoveries
        narratives_dir = os.path.join(base_dir, 'results', 'narratives')
        narrative_files = []
        if os.path.exists(narratives_dir):
            narrative_files = [f for f in os.listdir(narratives_dir) if f.endswith('.json')]
        
        # Get the next objectives in queue
        next_objectives = []
        try:
            # Try to get related objectives from the matrix
            related = self.objectives_manager.generate_related_objectives()
            if related:
                next_objectives = related[:3]  # Limit to 3 related objectives
        except:
            # If fails, leave empty
            pass
        
        # Build the summary
        summary_lines = [
            f"=== DISCOVERY SUMMARY: {timestamp} ===",
            f"Objective: {objective}",
            f"",
            f"Research Stats:",
            f"- URLs processed: {results['urls_processed']}",
            f"- Artifacts found: {results['artifacts_found']}",
            f"- High-scoring artifacts: {results['high_scoring_artifacts']}",
            f"- Narrative-worthy discoveries: {self.stats.get('narrative_worthy_discoveries', 0)}",
            f"- Research time: {self.stats.get('elapsed_minutes', 0):.2f} minutes",
            f"",
            f"Top Sources:"
        ]
        
        for domain, count in top_sources:
            summary_lines.append(f"- {domain}: {count} pages")
        
        summary_lines.append("")
        summary_lines.append("Discovery Highlights:")
        
        # Get the most recent narrative files (limit to 5)
        recent_narratives = sorted(narrative_files, reverse=True)[:5]
        narratives_data = []
        
        for narrative_file in recent_narratives:
            try:
                with open(os.path.join(narratives_dir, narrative_file), 'r') as f:
                    narrative_data = json.load(f)
                    if "discovery" in narrative_data:
                        narratives_data.append(narrative_data)
            except:
                continue
        
        # Add discovery highlights
        for i, narrative in enumerate(narratives_data):
            if "discovery" in narrative and "details" in narrative["discovery"]:
                details = narrative["discovery"]["details"]
                name = details.get('name', 'Unnamed')
                if name and len(name) > 30:
                    name = name[:27] + "..."
                
                summary_lines.append(
                    f"- {name}: "
                    f"Score {details.get('score', 'N/A')} | "
                    f"Type: {details.get('subtype', details.get('type', 'unknown'))} | "
                    f"Source: {urlparse(narrative['discovery'].get('url', '')).netloc}"
                )
        
        if not narratives_data:
            summary_lines.append("- No narrative-worthy discoveries in this session")
        
        # Add next objectives
        summary_lines.append("")
        summary_lines.append("Next Objectives in Queue:")
        if next_objectives:
            for i, obj in enumerate(next_objectives):
                summary_lines.append(f"- {obj}")
        else:
            summary_lines.append("- No specific objectives in queue")
        
        # Join all lines into a single string
        summary_text = "\n".join(summary_lines)
        
        # Save to session summaries directory
        summary_filename = f"{timestamp}.txt"
        summary_path = os.path.join(self.session_summaries_dir, summary_filename)
        
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        # Append to discovery log
        with open(self.discovery_log_path, 'a') as f:
            f.write(f"\n\n{summary_text}\n")
        
        # Print to console
        print("\n" + summary_text + "\n")
        
        logger.info(f"Discovery summary saved to {summary_path} and appended to discovery log")
        
        return summary_text
    
        main()