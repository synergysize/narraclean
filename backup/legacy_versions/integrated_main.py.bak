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
    
    def is_allowed_domain(self, url: str) -> bool:
        """
        Check if a domain is allowed for crawling.
        
        Args:
            url: URL to check
            
        Returns:
            Boolean indicating if domain is allowed
        """
        domain = urlparse(url).netloc
        
        # Allow any subdomain of allowed domains
        for allowed_domain in self.config.get("allowed_domains", []):
            if domain == allowed_domain or domain.endswith(f".{allowed_domain}"):
                return True
        
        return False
    
    def parse_objective(self, objective: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse an objective to extract artifact type and entity.
        
        Args:
            objective: Objective string
            
        Returns:
            Tuple of (artifact_type, entity)
        """
        if not objective:
            return None, None
        
        # Try to extract artifact type
        artifact_type = None
        for potential_type in ["name", "wallet", "code", "personal", "legal", "academic", "hidden", "institutional"]:
            if potential_type in objective.lower():
                artifact_type = potential_type
                break
        
        # Try to extract entity
        entity = None
        words = objective.split()
        for i, word in enumerate(words):
            if word.lower() in ["around", "related", "associated", "connected"] and i+1 < len(words):
                entity = " ".join(words[i+1:])
                # Remove any trailing punctuation
                entity = entity.rstrip(".,:;")
                break
        
        return artifact_type, entity
    
    def generate_research_urls(self, objective: str) -> List[str]:
        """
        Generate research URLs for an objective using LLM strategy.
        
        Args:
            objective: The research objective
            
        Returns:
            List of URLs to crawl
        """
        logger.info(f"Generating research URLs for objective: {objective}")
        
        # Parse objective to get artifact type and entity
        artifact_type, entity = self.parse_objective(objective)
        
        if not entity:
            logger.warning(f"Could not extract entity from objective: {objective}")
            return []
        
        # Use LLM to generate research strategy
        if self.config.get("llm_research_strategy", True):
            strategy = self.llm_strategy.generate_research_strategy(objective, entity)
            urls = strategy.get("crawlable_urls", [])
            
            logger.info(f"LLM strategy generated {len(urls)} URLs")
            
            # Add Wayback Machine URLs
            if self.config.get("wayback_integration", True):
                urls = self.wayback_machine.enrich_url_list_with_wayback(urls, objective)
                logger.info(f"Wayback integration added URLs, now have {len(urls)} URLs")
            
            return urls
        else:
            logger.info("LLM research strategy disabled, using basic URLs")
            
            # Basic URLs based on entity - removed hardcoded fallback URLs
            basic_urls = [
                f"https://duckduckgo.com/html/?q={entity.replace(' ', '+')}"
            ]
            
            return basic_urls
    
    def process_url(self, url: str, objective: str, depth: int) -> List[Dict[str, Any]]:
        """
        Process a URL for the current objective.
        
        Args:
            url: URL to process
            objective: Current research objective
            depth: Current crawl depth
            
        Returns:
            List of discovered artifacts
        """
        logger.info(f"Processing URL: {url} (depth: {depth})")
        
        # Parse objective to get artifact type and entity
        artifact_type, entity = self.parse_objective(objective)
        
        # Check if domain is allowed
        if not self.is_allowed_domain(url):
            logger.info(f"Skipping disallowed domain: {url}")
            return []
        
        # Check robots.txt
        if self.config.get("respect_robots", True) and not is_allowed_by_robots(url):
            logger.info(f"Skipping URL disallowed by robots.txt: {url}")
            return []
        
        # Fetch page
        try:
            html_content, fetch_info = fetch_page(url)
            self.stats["urls_processed"] += 1
            
            # Track domain for sources accessed
            domain = urlparse(url).netloc
            if "sources_accessed" not in self.stats:
                self.stats["sources_accessed"] = Counter()
            self.stats["sources_accessed"][domain] += 1
            
            # Track if this is a wayback snapshot
            if "web.archive.org" in url:
                if "wayback_snapshots_accessed" not in self.stats:
                    self.stats["wayback_snapshots_accessed"] = 0
                self.stats["wayback_snapshots_accessed"] += 1
                
            logger.info(f"Successfully fetched {url}")
            
            # Extract artifacts using the enhanced detector
            if self.config.get("enhanced_artifact_detection", True):
                artifacts = self.artifact_detector.extract_artifacts(
                    html_content, 
                    url=url, 
                    date=fetch_info.get("date"),
                    objective=objective,
                    entity=entity
                )
            else:
                # Fall back to standard extractor
                from artifact_extractor import extract_artifacts_from_html
                artifacts = extract_artifacts_from_html(
                    html_content, 
                    url=url, 
                    date=fetch_info.get("date")
                )
            
            self.stats["artifacts_found"] += len(artifacts)
            high_scoring = len([a for a in artifacts if a.get("score", 0) > 0.7])
            self.stats["high_scoring_artifacts"] += high_scoring
            
            if artifacts:
                logger.info(f"Found {len(artifacts)} artifacts ({high_scoring} high-scoring) on {url}")
            
            # Record discoveries in the matrix
            for artifact in artifacts:
                if artifact.get("score", 0) > 0.7:
                    # Prepare discovery for the matrix
                    discovery = {
                        "source": "crawler",
                        "url": url,
                        "content": artifact.get("summary", ""),
                        "entities": [entity] if entity else [],
                        "related_artifacts": [artifact_type] if artifact_type else [],
                        "details": artifact
                    }
                    
                    # Record in matrix
                    self.matrix.record_discovery(discovery, narrative_worthy=(artifact.get("score", 0) > 0.8))
                    
                    if artifact.get("score", 0) > 0.8:
                        self.stats["narrative_worthy_discoveries"] += 1
            
            # If we haven't reached max depth, extract and add links
            if depth < self.config.get("max_depth", 3):
                links = extract_links(url, html_content)
                
                # Filter links by allowed domains
                allowed_links = [link for link in links if self.is_allowed_domain(link)]
                
                # Add links to queue
                for link in allowed_links:
                    self.url_queue.add_url(link, depth + 1)
                
                logger.info(f"Added {len(allowed_links)} new URLs to queue from {url}")
            
            # Respect crawl delay
            time.sleep(self.config.get("crawl_delay", 2))
            
            return artifacts
        
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return []
    
    def process_objective(self, objective: str, max_urls: int = 100) -> Dict[str, Any]:
        """
        Process a research objective.
        
        Args:
            objective: The research objective
            max_urls: Maximum URLs to process for this objective
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Processing objective: {objective}")
        
        # Parse objective
        artifact_type, entity = self.parse_objective(objective)
        logger.info(f"Parsed objective - Artifact type: {artifact_type}, Entity: {entity}")
        
        # Reset URL queue
        self.url_queue = URLQueue()
        
        # Generate research URLs
        research_urls = self.generate_research_urls(objective)
        
        # Add URLs to queue
        for url in research_urls:
            self.url_queue.add_url(url, depth=0)
        
        logger.info(f"Added {len(research_urls)} research URLs to queue")
        
        # Process URLs
        processed_count = 0
        artifacts = []
        
        while not self.url_queue.is_empty() and processed_count < max_urls:
            url, depth = self.url_queue.next_url()
            
            if not url:
                break
            
            url_artifacts = self.process_url(url, objective, depth)
            artifacts.extend(url_artifacts)
            
            processed_count += 1
            
            # Check if we're making progress
            if len(artifacts) > 0 and len(artifacts) % 10 == 0:
                logger.info(f"Progress update: {len(artifacts)} artifacts found after processing {processed_count} URLs")
        
        # Return results
        results = {
            "objective": objective,
            "artifact_type": artifact_type,
            "entity": entity,
            "urls_processed": processed_count,
            "artifacts_found": len(artifacts),
            "high_scoring_artifacts": len([a for a in artifacts if a.get("score", 0) > 0.7])
        }
        
        logger.info(f"Objective processing complete: {results}")
        return results
    
    def run_with_objective(self, objective: str, max_time_minutes: int = 45) -> Dict[str, Any]:
        """
        Run the integrated controller with a specific objective.
        
        Args:
            objective: The research objective
            max_time_minutes: Maximum time to run in minutes
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Starting integrated controller with objective: {objective}")
        
        # Set the objective in the matrix
        with open(os.path.join(base_dir, 'results', 'current_objective.txt'), 'w') as f:
            f.write(objective)
        
        # Parse objective
        artifact_type, entity = self.parse_objective(objective)
        
        # Set up the objective in the matrix
        self.matrix.current_objective = {
            "text": objective,
            "artifact_type": artifact_type,
            "entity": entity,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "discoveries": []
        }
        
        # Initialize statistics
        self.stats = {
            "objectives_processed": 1,
            "urls_processed": 0,
            "artifacts_found": 0,
            "high_scoring_artifacts": 0,
            "narrative_worthy_discoveries": 0,
            "start_time": datetime.now(),
            "end_time": None
        }
        
        # Calculate end time
        end_time = datetime.now().timestamp() + (max_time_minutes * 60)
        
        # Process the objective
        max_urls_per_objective = self.config.get("max_pages_per_objective", 100)
        results = self.process_objective(objective, max_urls=max_urls_per_objective)
        
        # Check if we've run out of time
        if datetime.now().timestamp() > end_time:
            logger.info(f"Reached maximum time limit: {max_time_minutes} minutes")
        
        # Update stats
        self.stats["end_time"] = datetime.now()
        self.stats["elapsed_minutes"] = (self.stats["end_time"] - self.stats["start_time"]).total_seconds() / 60
        
        # Mark the objective as complete
        self.matrix.mark_objective_complete("completed")
        
        # Generate and save discovery summary
        summary = self.generate_discovery_summary(objective, results)
        
        return {
            "results": results,
            "stats": self.stats,
            "summary": summary
        }
    
    def run_autonomous(self, max_objectives: int = 3, max_time_minutes: int = 60) -> Dict[str, Any]:
        """
        Run the integrated controller autonomously, processing multiple objectives.
        
        Args:
            max_objectives: Maximum number of objectives to process
            max_time_minutes: Maximum total time to run in minutes
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Starting autonomous mode with max_objectives={max_objectives}, max_time={max_time_minutes}m")
        
        # Initialize statistics
        self.stats = {
            "objectives_processed": 0,
            "urls_processed": 0,
            "artifacts_found": 0,
            "high_scoring_artifacts": 0,
            "narrative_worthy_discoveries": 0,
            "start_time": datetime.now(),
            "end_time": None
        }
        
        # Calculate end time
        end_time = datetime.now().timestamp() + (max_time_minutes * 60)
        
        # Process objectives
        objectives_results = []
        
        for i in range(max_objectives):
            # Check if we've run out of time
            if datetime.now().timestamp() > end_time:
                logger.info(f"Reached maximum time limit: {max_time_minutes} minutes")
                break
            
            # Get the next objective
            objective = self.objectives_manager.load_current_objective()
            
            if not objective:
                logger.error("Failed to load or generate an objective")
                break
            
            logger.info(f"Processing objective {i+1}/{max_objectives}: {objective}")
            
            # Process the objective
            max_urls_per_objective = self.config.get("max_pages_per_objective", 100)
            results = self.process_objective(objective, max_urls=max_urls_per_objective)
            
            # Generate and save discovery summary
            summary = self.generate_discovery_summary(objective, results)
            results["summary"] = summary
            
            objectives_results.append(results)
            self.stats["objectives_processed"] += 1
            
            # Check if we should move to the next objective
            if self.objectives_manager.is_objective_exhausted():
                logger.info("Objective is exhausted, moving to next objective")
                
                # Generate related objectives
                related_objectives = self.objectives_manager.generate_related_objectives()
                for related in related_objectives:
                    logger.info(f"Generated related objective: {related}")
                
                # Move to the next objective
                next_objective = self.objectives_manager.move_to_next_objective()
                logger.info(f"Moved to next objective: {next_objective}")
        
        # Update stats
        self.stats["end_time"] = datetime.now()
        self.stats["elapsed_minutes"] = (self.stats["end_time"] - self.stats["start_time"]).total_seconds() / 60
        
        return {
            "objectives_results": objectives_results,
            "stats": self.stats
        }

def main():
    """Main entry point for the integrated controller."""
    parser = argparse.ArgumentParser(description="Integrated Narrahunt Phase 2 Controller")
    parser.add_argument("--objective", type=str, help="Specific objective to research")
    parser.add_argument("--time", type=int, default=45, help="Maximum time to run in minutes")
    parser.add_argument("--auto", action="store_true", help="Run in autonomous mode")
    parser.add_argument("--max-objectives", type=int, default=3, help="Maximum objectives to process in autonomous mode")
    
    args = parser.parse_args()
    
    # Initialize controller
    controller = IntegratedController()
    
    # Print header
    print("=" * 80)
    print("Narrahunt Phase 2 - Integrated Controller")
    print("=" * 80)
    
    if args.auto:
        print(f"Running in autonomous mode with max_objectives={args.max_objectives}, max_time={args.time}m")
        results = controller.run_autonomous(max_objectives=args.max_objectives, max_time_minutes=args.time)
        
        print("\nAutonomous Mode Results:")
        print(f"Processed {results['stats']['objectives_processed']} objectives")
        print(f"Total URLs processed: {results['stats']['urls_processed']}")
        print(f"Total artifacts found: {results['stats']['artifacts_found']}")
        print(f"High-scoring artifacts: {results['stats']['high_scoring_artifacts']}")
        print(f"Narrative-worthy discoveries: {results['stats']['narrative_worthy_discoveries']}")
        print(f"Total elapsed time: {results['stats']['elapsed_minutes']:.2f} minutes")
        
        print("\nObjectives Processed:")
        for i, obj_result in enumerate(results['objectives_results']):
            print(f"{i+1}. {obj_result['objective']}")
            print(f"   - URLs processed: {obj_result['urls_processed']}")
            print(f"   - Artifacts found: {obj_result['artifacts_found']}")
            print(f"   - High-scoring artifacts: {obj_result['high_scoring_artifacts']}")
    
    elif args.objective:
        print(f"Running with specific objective: {args.objective}")
        results = controller.run_with_objective(args.objective, max_time_minutes=args.time)
        
        print("\nResults:")
        print(f"Objective: {results['results']['objective']}")
        print(f"URLs processed: {results['results']['urls_processed']}")
        print(f"Artifacts found: {results['results']['artifacts_found']}")
        print(f"High-scoring artifacts: {results['results']['high_scoring_artifacts']}")
        print(f"Narrative-worthy discoveries: {results['stats']['narrative_worthy_discoveries']}")
        print(f"Elapsed time: {results['stats']['elapsed_minutes']:.2f} minutes")
        
        # Check narratives directory
        narratives_dir = os.path.join(base_dir, 'results', 'narratives')
        narrative_files = [f for f in os.listdir(narratives_dir) if f.endswith('.json')]
        
        if narrative_files:
            print("\nNarrative-worthy discoveries saved to:")
            for file in narrative_files[-5:]:  # Show last 5 files
                print(f"- {os.path.join('results', 'narratives', file)}")
    
    else:
        print("No objective specified. Please provide --objective or use --auto mode.")
        parser.print_help()
    
    print("\nController execution complete.")

if __name__ == "__main__":
    main()