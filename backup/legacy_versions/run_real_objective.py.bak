#!/usr/bin/env python3
"""
Run a real objective with the fixed system.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# Set up base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_dir, 'logs', 'real_objective.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('real_objective')

# Import components
from narrative_matrix import NarrativeMatrix
from objectives_manager import ObjectivesManager
from llm_research_strategy import LLMResearchStrategy
from crawler import Crawler
from llm_integration import LLMIntegration
from enhancements.name_artifact_extractor import NameArtifactExtractor

def run_objective(objective: str, max_urls: int = 10, max_time_minutes: int = 45):
    """
    Run a specific objective with the fixed system.
    
    Args:
        objective: The objective to run
        max_urls: Maximum number of URLs to process
        max_time_minutes: Maximum time to run in minutes
    """
    logger.info(f"Running objective: {objective}")
    logger.info(f"Max URLs: {max_urls}, Max time: {max_time_minutes} minutes")
    
    # Initialize components
    matrix = NarrativeMatrix()
    llm = LLMIntegration(use_claude=True)
    strategy_generator = LLMResearchStrategy()
    crawler = Crawler()
    
    # Parse objective
    words = objective.split()
    artifact_type = None
    entity = None
    
    for i, word in enumerate(words):
        if word.lower() in ["find", "discover"] and i+1 < len(words):
            artifact_type = words[i+1]
        if word.lower() in ["around", "related", "associated", "connected"] and i+1 < len(words):
            entity = " ".join(words[i+1:])
            entity = entity.rstrip(".,:;")
    
    logger.info(f"Parsed objective - Artifact type: {artifact_type}, Entity: {entity}")
    
    # Set up current objective in matrix
    with open(os.path.join(base_dir, 'results', 'current_objective.txt'), 'w') as f:
        f.write(objective)
    
    matrix.current_objective = {
        "text": objective,
        "artifact_type": artifact_type,
        "entity": entity,
        "created_at": datetime.now().isoformat(),
        "status": "active",
        "discoveries": []
    }
    
    # Get research strategy
    print("Generating research strategy...")
    strategy = strategy_generator.generate_research_strategy(objective, entity)
    
    urls_to_crawl = strategy["crawlable_urls"][:max_urls]
    print(f"Generated {len(urls_to_crawl)} URLs to crawl")
    
    # Create specialized extractor if needed
    specialized_extractor = None
    if artifact_type == "name":
        specialized_extractor = NameArtifactExtractor(entity=entity)
        print("Using specialized name artifact extractor")
    
    # Crawl URLs
    print("Starting crawl...")
    all_artifacts = []
    narrative_worthy_count = 0
    
    for i, url in enumerate(urls_to_crawl):
        print(f"Processing URL {i+1}/{len(urls_to_crawl)}: {url}")
        
        html_content, artifacts = crawler.process_url(url, depth=0, extract_links_flag=False)
        
        if html_content and specialized_extractor:
            # Extract with specialized extractor
            specialized_artifacts = specialized_extractor.extract_from_html(html_content, url=url)
            
            for artifact in specialized_artifacts:
                # Record in matrix
                discovery = {
                    "source": "specialized_extractor",
                    "url": url,
                    "content": f"{artifact_type.title()} artifact: {artifact['name']} ({artifact['subtype']})",
                    "entities": [entity] if entity else [],
                    "related_artifacts": [artifact_type] if artifact_type else [],
                    "details": artifact
                }
                
                # Determine if narrative-worthy
                is_narrative = artifact.get("score", 0) > 0.8
                
                # Record in matrix
                matrix.record_discovery(discovery, narrative_worthy=is_narrative)
                
                if is_narrative:
                    narrative_worthy_count += 1
                
                all_artifacts.append(artifact)
        
        elif html_content:
            # Use standard artifacts
            for artifact in artifacts:
                if artifact.get("score", 0) > 0.7:
                    # Record in matrix
                    discovery = {
                        "source": "crawler",
                        "url": url,
                        "content": artifact.get("summary", ""),
                        "entities": [entity] if entity else [],
                        "related_artifacts": [artifact_type] if artifact_type else [],
                        "details": artifact
                    }
                    
                    # Determine if narrative-worthy
                    is_narrative = artifact.get("score", 0) > 0.8
                    
                    # Record in matrix
                    matrix.record_discovery(discovery, narrative_worthy=is_narrative)
                    
                    if is_narrative:
                        narrative_worthy_count += 1
            
            all_artifacts.extend(artifacts)
    
    # Mark objective as complete
    matrix.mark_objective_complete("completed")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("OBJECTIVE RESULTS SUMMARY")
    print("=" * 80)
    print(f"Objective: {objective}")
    print(f"URLs processed: {len(urls_to_crawl)}")
    print(f"Total artifacts found: {len(all_artifacts)}")
    print(f"Narrative-worthy discoveries: {narrative_worthy_count}")
    
    # Check narratives directory
    narratives_dir = os.path.join(base_dir, 'results', 'narratives')
    narrative_files = []
    if os.path.exists(narratives_dir):
        narrative_files = [f for f in os.listdir(narratives_dir) if f.endswith('.json')]
    
    print("\nNarrative Discoveries:")
    for i, narrative_file in enumerate(sorted(narrative_files)[-5:]):  # Show last 5
        try:
            with open(os.path.join(narratives_dir, narrative_file), 'r') as f:
                narrative = json.load(f)
                
                if "discovery" in narrative and "details" in narrative["discovery"]:
                    details = narrative["discovery"]["details"]
                    print(f"{i+1}. {details.get('name', 'Unnamed')}")
                    print(f"   Type: {details.get('subtype', details.get('type', 'unknown'))}")
                    print(f"   Score: {details.get('score', 'N/A')}")
                    print(f"   Source: {narrative['discovery'].get('url', '')}")
        except:
            continue
    
    print("\nObjective completed successfully.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run a real objective with the fixed system")
    parser.add_argument("--objective", type=str, default="Find name artifacts around Vitalik Buterin",
                      help="The objective to run")
    parser.add_argument("--max-urls", type=int, default=10, help="Maximum number of URLs to process")
    parser.add_argument("--max-time", type=int, default=45, help="Maximum time to run in minutes")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RUNNING REAL OBJECTIVE")
    print("=" * 80)
    
    run_objective(args.objective, args.max_urls, args.max_time)

if __name__ == "__main__":
    main()