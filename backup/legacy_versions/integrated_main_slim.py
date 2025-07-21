#!/usr/bin/env python3
"""
Integrated Main Controller for Narrahunt Phase 2.

This slim version imports core functionality from the modules directory.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core functionality from modules
from modules.routing import review_research_strategy
from modules.llm_engine import get_llm_instance
from modules.utils import is_allowed_by_robots, is_allowed_domain

class IntegratedController:
    """
    Integrated Main Controller for Narrahunt Phase 2.
    
    This controller integrates the Narrative Discovery Matrix with the existing
    crawler infrastructure, using matrix-generated objectives to guide research.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the integrated controller.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.discovery_path = self.config.get("discovery_path", "discoveries")
        self.discovery_log_path = os.path.join(self.discovery_path, "discovery_log.txt")
        
        # Create discovery directory if it doesn't exist
        os.makedirs(self.discovery_path, exist_ok=True)
        
        logger.info(f"IntegratedController initialized with config: {config_path}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            "discovery_path": "discoveries",
            "max_depth": 3,
            "allowed_domains": [
                "ethereum.org",
                "vitalik.ca",
                "github.com",
                "ethresear.ch",
                "ethhub.io"
            ],
            "use_wayback": True,
            "use_llm_analysis": True
        }
        
        if not config_path or not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Merge with defaults
            merged_config = {**default_config, **config}
            return merged_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def run_investigation(self, objective: str, entity: str, max_iterations: int = 50):
        """
        Run a full investigation using the detective agent.
        
        Args:
            objective: The research objective
            entity: The primary entity to investigate
            max_iterations: Maximum iterations to run
            
        Returns:
            Path to the generated report
        """
        from detective_agent_slim import DetectiveAgent
        
        # Create and run detective agent
        detective = DetectiveAgent(
            objective=objective,
            entity=entity,
            max_iterations=max_iterations,
            save_path=self.discovery_path
        )
        
        # Run the investigation
        discoveries = detective.start_investigation()
        
        # Generate report path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.discovery_path, f"report_{timestamp}.md")
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(f"# Investigation Report: {objective}\n\n")
            f.write(f"## Primary Entity: {entity}\n\n")
            f.write(f"## Discoveries ({len(discoveries)})\n\n")
            
            for i, discovery in enumerate(discoveries, 1):
                f.write(f"### {i}. {discovery.get('title', 'Untitled Discovery')}\n\n")
                f.write(f"**Source:** {discovery.get('url', 'Unknown')}\n\n")
                f.write(f"**Date:** {discovery.get('date', 'Unknown')}\n\n")
                f.write(f"**Content:**\n\n{discovery.get('content', '')}\n\n")
                f.write("---\n\n")
                
        logger.info(f"Investigation report saved to: {report_path}")
        return report_path

def main():
    """Main function to run the integrated controller."""
    parser = argparse.ArgumentParser(description='Run an integrated investigation')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--objective', '-o', default="Investigate the early history of Ethereum", help='Research objective')
    parser.add_argument('--entity', '-e', default="Ethereum", help='Primary entity to investigate')
    parser.add_argument('--iterations', '-i', type=int, default=50, help='Maximum iterations')
    
    args = parser.parse_args()
    
    # Create and run controller
    controller = IntegratedController(args.config)
    report_path = controller.run_investigation(args.objective, args.entity, args.iterations)
    
    print(f"Investigation complete. Report saved to: {report_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())