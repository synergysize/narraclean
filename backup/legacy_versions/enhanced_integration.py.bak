#!/usr/bin/env python3
"""
Enhanced Integration for the Narrative Discovery Matrix with Name Artifact Extractor
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

# Import the narrative matrix and enhanced extractor
from narrative_matrix import NarrativeMatrix
from objectives_manager import ObjectivesManager
from enhancements.name_artifact_extractor import NameArtifactExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_dir, 'logs', 'enhanced_integration.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('enhanced_integration')

class EnhancedIntegration:
    """
    Enhanced integration of the Narrative Discovery Matrix with specialized extractors.
    """
    
    def __init__(self):
        """Initialize the enhanced integration."""
        self.matrix = NarrativeMatrix()
        self.manager = ObjectivesManager()
        
        # Create directories
        os.makedirs(os.path.join(project_dir, 'results', 'enhanced'), exist_ok=True)
    
    def run_name_focused_objective(self, entity, max_time_minutes=30):
        """
        Run a name-focused objective for a specific entity.
        
        Args:
            entity: The entity to focus on
            max_time_minutes: Maximum time to run in minutes
        """
        logger.info(f"Starting name-focused objective for entity: {entity}")
        
        # Ensure entity is in the matrix
        if entity not in self.matrix.config["specific_targets"]:
            self.matrix.add_entity(entity, "specific")
        
        # Ensure "name" is in the artifact types
        if "name" not in self.matrix.config["artifact_types"]:
            self.matrix.add_artifact_type("name")
        
        # Create and set the objective
        objective = f"Find name around {entity}"
        with open(os.path.join(project_dir, 'results', 'current_objective.txt'), 'w') as f:
            f.write(objective)
        
        # Set as current objective in the matrix
        self.matrix.current_objective = {
            "text": objective,
            "artifact_type": "name",
            "entity": entity,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "discoveries": []
        }
        
        logger.info(f"Set objective: {objective}")
        
        # Create specialized name artifact extractor
        name_extractor = NameArtifactExtractor(entity=entity)
        
        # Run test with sample data that has specific formatting for matching patterns
        sample_text = """
        Vitalik Buterin (username: vitalik_btc on the Bitcoin forum) is the creator of Ethereum.
        Before creating Ethereum, he worked on a project called "Colored Coins" and contributed to Bitcoin Magazine.
        His early pseudonym was "Bitcoinmeister" in some communities.
        The Ethereum project was initially called "Frontier" during its first release phase.
        He founded the Ethereum Foundation to support development of the blockchain.
        Vitalik coined the term "smart contract" to describe the programmable features of Ethereum.
        His GitHub handle is "vbuterin" where he commits code for various projects.
        """
        
        artifacts = name_extractor.extract_from_text(sample_text, url="https://example.com")
        
        # Record discoveries in the matrix
        for artifact in artifacts:
            discovery = {
                "source": "name_extractor",
                "url": artifact.get("source_url", ""),
                "content": f"Name artifact: {artifact['name']} ({artifact['subtype']})",
                "entities": [entity],
                "related_artifacts": ["name"],
                "details": {
                    "subtype": artifact["subtype"],
                    "context": artifact["context"],
                    "score": artifact["score"]
                }
            }
            
            # Determine if it's narrative-worthy based on score
            narrative_worthy = artifact["score"] > 0.7
            self.matrix.record_discovery(discovery, narrative_worthy=narrative_worthy)
        
        logger.info(f"Recorded {len(artifacts)} name artifacts for {entity}")
        
        # Save artifacts to enhanced results directory
        output_dir = os.path.join(project_dir, 'results', 'enhanced', 'name_artifacts')
        name_extractor.save_artifacts(artifacts, output_dir)
        
        # Mark the objective as complete
        self.matrix.mark_objective_complete("completed")
        logger.info(f"Completed name-focused objective for {entity}")
        
        return artifacts

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(project_dir, 'logs'), exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run enhanced integration for the Narrative Discovery Matrix')
    parser.add_argument('--entity', type=str, default="Vitalik Buterin", 
                        help='Entity to focus on')
    parser.add_argument('--time', type=int, default=30, 
                        help='Maximum time to run in minutes')
    args = parser.parse_args()
    
    # Run the enhanced integration
    integration = EnhancedIntegration()
    
    print("=" * 80)
    print(f"Enhanced Integration: Name-Focused Objective for {args.entity}")
    print("=" * 80)
    
    artifacts = integration.run_name_focused_objective(args.entity, args.time)
    
    print(f"\nFound {len(artifacts)} name artifacts for {args.entity}:")
    for i, artifact in enumerate(artifacts):
        print(f"{i+1}. {artifact['name']} ({artifact['subtype']}, score: {artifact['score']})")
    
    print("\nComplete results saved to:")
    print(f"- {os.path.join(project_dir, 'results', 'enhanced', 'name_artifacts')}")
    
    print("\nIntegration complete.")