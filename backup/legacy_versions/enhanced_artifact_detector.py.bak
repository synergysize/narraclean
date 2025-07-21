#!/usr/bin/env python3
"""
Enhanced Artifact Detector for Narrahunt Phase 2.

This module routes content through specialized extractors based on
the current objective and maintains a registry of extractors.
"""

import os
import sys
import logging
import importlib
from typing import List, Dict, Any, Optional, Callable, Type

# Set up base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Configure logging
os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_dir, 'logs', 'enhanced_detector.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('enhanced_artifact_detector')

# Import standard artifact extractor
from core.artifact_extractor import extract_artifacts_from_html

# Import any specialized extractors if available
try:
    from enhancements.name_artifact_extractor import NameArtifactExtractor
    name_extractor_available = True
    logger.info("Name artifact extractor is available")
except ImportError:
    name_extractor_available = False
    logger.warning("Name artifact extractor is not available")

class EnhancedArtifactDetector:
    """
    Routes content through specialized extractors based on the current objective.
    """
    
    def __init__(self):
        """Initialize the enhanced artifact detector."""
        self.extractors = {}
        self.register_default_extractors()
    
    def register_default_extractors(self):
        """Register the default extractors."""
        # Register the standard extractor for all artifact types
        self.register_extractor('default', extract_artifacts_from_html)
        
        # Register specialized extractors if available
        if name_extractor_available:
            self.register_extractor('name', self._extract_name_artifacts)
    
    def register_extractor(self, artifact_type: str, extractor_function: Callable):
        """
        Register an extractor for a specific artifact type.
        
        Args:
            artifact_type: The type of artifact this extractor handles
            extractor_function: The extractor function
        """
        self.extractors[artifact_type] = extractor_function
        logger.info(f"Registered extractor for artifact type: {artifact_type}")
    
    def get_extractor(self, artifact_type: str) -> Callable:
        """
        Get the appropriate extractor for an artifact type.
        
        Args:
            artifact_type: The type of artifact to extract
            
        Returns:
            The extractor function
        """
        if artifact_type in self.extractors:
            return self.extractors[artifact_type]
        else:
            return self.extractors['default']
    
    def _extract_name_artifacts(self, html_content: str, url: str = "", date: str = None, entity: str = None) -> List[Dict[str, Any]]:
        """
        Extract name artifacts using the specialized name extractor.
        
        Args:
            html_content: The HTML content to analyze
            url: The source URL
            date: The date of the content
            entity: The target entity (optional)
            
        Returns:
            List of artifact dictionaries
        """
        # Create a name extractor instance with target entity for auto-exclusion
        name_extractor = NameArtifactExtractor(entity=entity)
        
        # Extract artifacts
        artifacts = name_extractor.extract_from_html(html_content, url=url, date=date)
        
        # Unique artifacts set for deduplication
        unique_names = set()
        
        # Convert to standard format
        standardized_artifacts = []
        
        for artifact in artifacts:
            name = artifact.get("name", "")
            name_lower = name.lower()
            
            # Skip if no name or if it's too short
            if not name or len(name) < 2:
                continue
                
            # Skip duplicates
            if name_lower in unique_names:
                logger.debug(f"Skipping duplicate name artifact: {name}")
                continue
                
            # Skip if name is just the entity or part of entity name
            if entity and (name_lower == entity.lower() or entity.lower().startswith(name_lower + " ")):
                logger.debug(f"Skipping entity name artifact: {name}")
                continue
            
            # Add to unique set
            unique_names.add(name_lower)
            
            # Generate a simple hash for the artifact based on name and source
            artifact_hash = str(hash(f"{name_lower}_{url}"))
            
            # Adjust score - ensure valuable artifacts are scored higher
            score = artifact.get("score", 0.5)
            
            # Boost score for likely valuable artifacts
            if artifact.get("subtype") == "username" and re.match(r'^[a-zA-Z0-9_-]+$', name):
                score = min(1.0, score + 0.1)  # Usernames are valuable
                
            # Standardize the artifact
            standardized = {
                "type": "name",
                "subtype": artifact.get("subtype", "unknown"),
                "content": artifact.get("name", ""),
                "summary": f"Name artifact: {name} ({artifact.get('subtype', 'unknown')})",
                "location": "HTML content",
                "hash": artifact_hash,
                "score": score,
                "url": url,
                "date": date,
                "entity": entity,
                "name": name
            }
            standardized_artifacts.append(standardized)
        
        # Log summary
        logger.info(f"Extracted {len(standardized_artifacts)} unique name artifacts from URL: {url}")
        
        # Sort by score (descending) and return
        return sorted(standardized_artifacts, key=lambda x: x.get("score", 0), reverse=True)
    
    def extract_artifacts(self, html_content: str, url: str, date: Optional[str] = None, 
                         objective: Optional[str] = None, entity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract artifacts using the appropriate extractor based on the objective.
        
        Args:
            html_content: The HTML content to analyze
            url: The source URL
            date: The date of the content
            objective: The current research objective
            entity: The target entity
            
        Returns:
            List of artifact dictionaries
        """
        # Determine artifact type from objective
        artifact_type = 'default'
        if objective:
            for potential_type in ["name", "wallet", "code", "personal", "legal", "academic", "hidden", "institutional"]:
                if potential_type in objective.lower():
                    artifact_type = potential_type
                    break
        
        logger.info(f"Using {artifact_type} extractor for URL: {url}")
        
        # Get the appropriate extractor
        extractor = self.get_extractor(artifact_type)
        
        # Extract artifacts
        if artifact_type == 'name':
            artifacts = extractor(html_content, url, date, entity)
        else:
            artifacts = extractor(html_content, url, date)
        
        # Apply objective-specific scoring
        artifacts = self._apply_objective_scoring(artifacts, objective, url)
        
        return artifacts
    
    def _apply_objective_scoring(self, artifacts: List[Dict[str, Any]], 
                               objective: Optional[str], url: str) -> List[Dict[str, Any]]:
        """
        Apply objective-specific scoring adjustments to artifacts.
        
        Args:
            artifacts: List of extracted artifacts
            objective: The current research objective
            url: The source URL
            
        Returns:
            List of artifacts with adjusted scores
        """
        if not objective:
            return artifacts
        
        for artifact in artifacts:
            # Base score from extractor
            base_score = artifact.get("score", 0.5)
            
            # Adjust score based on objective match
            artifact_type = artifact.get("type", "")
            if artifact_type in objective.lower():
                # Boost score for artifacts matching the objective type
                artifact["score"] = min(1.0, base_score + 0.2)
            
            # Adjust based on source relevance
            if "vitalik.ca" in url and "vitalik" in objective.lower():
                artifact["score"] = min(1.0, artifact["score"] + 0.1)
            elif "github.com/vbuterin" in url and "vitalik" in objective.lower():
                artifact["score"] = min(1.0, artifact["score"] + 0.1)
            elif "ethereum.org" in url and "ethereum" in objective.lower():
                artifact["score"] = min(1.0, artifact["score"] + 0.1)
            elif "web.archive.org" in url and any(domain in url for domain in ["vitalik.ca", "ethereum.org"]):
                artifact["score"] = min(1.0, artifact["score"] + 0.05)
        
        return artifacts

if __name__ == "__main__":
    # Test the enhanced artifact detector
    detector = EnhancedArtifactDetector()
    
    # Sample HTML content
    sample_html = """
    <html>
    <head><title>Vitalik Buterin - Ethereum Creator</title></head>
    <body>
        <h1>About Vitalik Buterin</h1>
        <p>Vitalik Buterin (username: vitalik_btc on the Bitcoin forum) is the creator of Ethereum.</p>
        <p>Before creating Ethereum, he worked on a project called "Colored Coins" and contributed to Bitcoin Magazine.</p>
        <p>His early pseudonym was "Bitcoinmeister" in some communities.</p>
        <p>The Ethereum project was initially called "Frontier" during its first release phase.</p>
        <p>He founded the Ethereum Foundation to support development of the blockchain.</p>
        <p>His GitHub handle is "vbuterin" where he commits code for various projects.</p>
    </body>
    </html>
    """
    
    # Test extraction with different objectives
    test_objectives = [
        "Find name artifacts around Vitalik Buterin",
        "Find wallet artifacts around Vitalik Buterin",
        "Find code artifacts around Ethereum developers"
    ]
    
    for objective in test_objectives:
        print(f"\nTesting with objective: {objective}")
        artifacts = detector.extract_artifacts(
            sample_html, 
            url="https://example.com/vitalik", 
            date="2023-07-15",
            objective=objective,
            entity="Vitalik Buterin"
        )
        
        print(f"Found {len(artifacts)} artifacts")
        for i, artifact in enumerate(artifacts):
            print(f"{i+1}. {artifact.get('summary', 'No summary')} (Score: {artifact.get('score', 'N/A')})")