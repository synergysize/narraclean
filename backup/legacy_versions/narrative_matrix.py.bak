#!/usr/bin/env python3
"""
Narrative Discovery Matrix System
---------------------------------
This module defines the core matrix system that generates and manages
objectives for narrative discovery.
"""

import json
import os
import random
from typing import List, Dict, Any
import logging
from datetime import datetime
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'matrix.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('narrative_matrix')

class NarrativeMatrix:
    """
    Generates and manages narrative discovery objectives by combining
    artifact types with target entities.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the narrative matrix system."""
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config', 'narrative_matrix.json'
        )
        self.current_objective_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'results', 'current_objective.txt'
        )
        self.narratives_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'results', 'narratives'
        )
        
        # Create necessary directories if they don't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.current_objective_path), exist_ok=True)
        os.makedirs(self.narratives_dir, exist_ok=True)
        
        # Load or initialize configuration
        self.config = self._load_or_init_config()
        
        # Track generated objectives to avoid duplicates
        self.generated_objectives = set()
        self.current_objective = None
        
    def _load_or_init_config(self) -> Dict[str, Any]:
        """Load existing configuration or initialize a new one."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error loading config from {self.config_path}. Creating new config.")
        
        # Default configuration
        default_config = {
            "artifact_types": [
                "code", "wallet", "name", "personal", "legal", 
                "academic", "hidden", "institutional"
            ],
            "target_entities": [
                "crypto devs", "meme creators", "companies", "institutions"
            ],
            "specific_targets": [],  # Will be populated as discoveries are made
            "objective_templates": [
                "Find {artifact_type} around {entity}",
                "Discover {artifact_type} related to {entity}",
                "Research {artifact_type} associated with {entity}",
                "Investigate {artifact_type} connected to {entity}"
            ],
            "progress_thresholds": {
                "dead_end_time_minutes": 30,
                "min_discoveries_for_success": 2
            }
        }
        
        # Save the default configuration
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def save_config(self) -> None:
        """Save the current configuration to disk using atomic write."""
        temp_path = self.config_path + '.tmp'
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            shutil.move(temp_path, self.config_path)
            logger.info("Configuration saved successfully.")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Error saving configuration: {e}")
            raise e
    
    def add_entity(self, entity: str, entity_type: str = None) -> None:
        """
        Add a new entity to the target list.
        
        Args:
            entity: The entity to add
            entity_type: Optional type classification
        """
        if entity_type and entity_type == "specific":
            if entity not in self.config["specific_targets"]:
                self.config["specific_targets"].append(entity)
                logger.info(f"Added specific target: {entity}")
        else:
            if entity not in self.config["target_entities"]:
                self.config["target_entities"].append(entity)
                logger.info(f"Added target entity: {entity}")
        
        self.save_config()
    
    def add_artifact_type(self, artifact_type: str) -> None:
        """Add a new artifact type to the matrix."""
        if artifact_type not in self.config["artifact_types"]:
            self.config["artifact_types"].append(artifact_type)
            logger.info(f"Added artifact type: {artifact_type}")
            self.save_config()
    
    def generate_objective(self) -> str:
        """
        Generate a new objective by combining an artifact type with a target entity.
        
        Returns:
            A formatted objective string
        """
        # Combine regular entities and specific targets
        all_entities = self.config["target_entities"] + self.config["specific_targets"]
        
        # Safety check for empty lists
        if not all_entities:
            logger.error("No entities available for objective generation")
            return "No entities configured for research"
        
        if not self.config["artifact_types"]:
            logger.error("No artifact types available for objective generation")
            return "No artifact types configured for research"
        
        # Try to find an unused combination
        max_attempts = len(all_entities) * len(self.config["artifact_types"])
        attempts = 0
        
        while attempts < max_attempts:
            artifact_type = random.choice(self.config["artifact_types"])
            entity = random.choice(all_entities)
            template = random.choice(self.config["objective_templates"])
            
            objective = template.format(artifact_type=artifact_type, entity=entity)
            
            if objective not in self.generated_objectives:
                self.generated_objectives.add(objective)
                self.current_objective = {
                    "text": objective,
                    "artifact_type": artifact_type,
                    "entity": entity,
                    "created_at": datetime.now().isoformat(),
                    "status": "active",
                    "discoveries": []
                }
                
                # Save the current objective using atomic write
                temp_path = self.current_objective_path + '.tmp'
                try:
                    with open(temp_path, 'w') as f:
                        f.write(objective)
                    shutil.move(temp_path, self.current_objective_path)
                    logger.info(f"Generated new objective: {objective}")
                except Exception as e:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    logger.error(f"Error saving objective: {e}")
                    raise e
                return objective
            
            attempts += 1
        
        # If we couldn't find an unused combination
        logger.warning("All possible objective combinations have been generated.")
        return "All possible objectives have been exhausted"
    
    def record_discovery(self, discovery: Dict[str, Any], narrative_worthy: bool = False) -> None:
        """
        Record a discovery related to the current objective.
        
        Args:
            discovery: Dictionary containing discovery details
            narrative_worthy: Whether this discovery should be logged as a narrative
        """
        if not self.current_objective:
            logger.error("Cannot record discovery: No active objective")
            return
        
        # Add timestamp if not present
        if "timestamp" not in discovery:
            discovery["timestamp"] = datetime.now().isoformat()
        
        # Add to current objective discoveries
        self.current_objective["discoveries"].append(discovery)
        
        # Extract potential new entities or artifacts
        if "entities" in discovery:
            for entity in discovery["entities"]:
                self.add_entity(entity, "specific")
        
        if narrative_worthy:
            self._log_narrative(discovery)
    
    def _log_narrative(self, discovery: Dict[str, Any]) -> None:
        """Log a narrative-worthy discovery to the narratives directory."""
        artifact_type = self.current_objective["artifact_type"]
        
        # Create a filename based on timestamp and artifact type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{artifact_type}.json"
        filepath = os.path.join(self.narratives_dir, filename)
        
        narrative_data = {
            "objective": self.current_objective["text"],
            "artifact_type": artifact_type,
            "entity": self.current_objective["entity"],
            "discovery": discovery,
            "timestamp": datetime.now().isoformat()
        }
        
        temp_path = filepath + '.tmp'
        try:
            with open(temp_path, 'w') as f:
                json.dump(narrative_data, f, indent=2)
            shutil.move(temp_path, filepath)
            logger.info(f"Logged narrative-worthy discovery to {filepath}")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Error logging narrative discovery: {e}")
            raise e
    
    def mark_objective_complete(self, status: str = "completed") -> None:
        """
        Mark the current objective as complete.
        
        Args:
            status: Status to set (completed, dead_end, etc.)
        """
        if not self.current_objective:
            logger.error("No active objective to mark complete")
            return
        
        self.current_objective["status"] = status
        self.current_objective["completed_at"] = datetime.now().isoformat()
        
        # Archive the completed objective
        archive_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'completed_objectives')
        os.makedirs(archive_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{status}.json"
        filepath = os.path.join(archive_dir, filename)
        
        temp_path = filepath + '.tmp'
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.current_objective, f, indent=2)
            shutil.move(temp_path, filepath)
            logger.info(f"Marked objective as {status} and archived to {filepath}")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Error archiving objective: {e}")
            raise e
        
        # Clear the current objective
        self.current_objective = None
        if os.path.exists(self.current_objective_path):
            os.remove(self.current_objective_path)
    
    def is_dead_end(self) -> bool:
        """
        Check if the current objective has hit a dead end based on time
        since last discovery or other criteria.
        
        Returns:
            Boolean indicating if the objective is at a dead end
        """
        if not self.current_objective:
            return False
        
        # Check if there are any discoveries
        if not self.current_objective["discoveries"]:
            # Check time since creation
            created_time = datetime.fromisoformat(self.current_objective["created_at"])
            elapsed_minutes = (datetime.now() - created_time).total_seconds() / 60
            
            if elapsed_minutes > self.config["progress_thresholds"]["dead_end_time_minutes"]:
                logger.info(f"Objective hit dead end: No discoveries after {elapsed_minutes:.1f} minutes")
                return True
        else:
            # Check time since last discovery
            last_discovery_time = datetime.fromisoformat(
                self.current_objective["discoveries"][-1]["timestamp"]
            )
            elapsed_minutes = (datetime.now() - last_discovery_time).total_seconds() / 60
            
            if elapsed_minutes > self.config["progress_thresholds"]["dead_end_time_minutes"]:
                logger.info(f"Objective hit dead end: No new discoveries after {elapsed_minutes:.1f} minutes")
                return True
        
        return False
    
    def generate_followup_objectives(self) -> List[str]:
        """
        Generate follow-up objectives based on discoveries in the current objective.
        
        Returns:
            List of follow-up objective strings
        """
        if not self.current_objective or not self.current_objective["discoveries"]:
            logger.warning("Cannot generate follow-ups: No active objective or discoveries")
            return []
        
        followups = []
        
        # Extract entities from discoveries
        new_entities = []
        for discovery in self.current_objective["discoveries"]:
            if "entities" in discovery:
                new_entities.extend(discovery["entities"])
            if "related_artifacts" in discovery:
                for artifact in discovery["related_artifacts"]:
                    if artifact not in self.config["artifact_types"]:
                        self.add_artifact_type(artifact)
        
        # Add new entities to targets and generate objectives for them
        for entity in new_entities:
            if entity not in self.config["specific_targets"]:
                self.add_entity(entity, "specific")
                
                # Generate a follow-up objective for this entity
                template = random.choice(self.config["objective_templates"])
                artifact_type = random.choice(self.config["artifact_types"])
                
                followup = template.format(artifact_type=artifact_type, entity=entity)
                followups.append(followup)
        
        logger.info(f"Generated {len(followups)} follow-up objectives")
        return followups

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)
    
    # Test the narrative matrix
    matrix = NarrativeMatrix()
    objective = matrix.generate_objective()
    print(f"Generated objective: {objective}")
    
    # Example discovery
    matrix.record_discovery({
        "source": "example_crawler",
        "content": "Found GitHub repository with personal information",
        "entities": ["John Doe", "CryptoProject X"],
        "related_artifacts": ["repository", "social_media"]
    }, narrative_worthy=True)
    
    print("Recorded test discovery and generated follow-up objectives")