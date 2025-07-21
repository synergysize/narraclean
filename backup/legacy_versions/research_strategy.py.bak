#!/usr/bin/env python3
"""
Research Strategy for Detective Agent.

This module implements a methodical, depth-first exploration strategy 
for the Detective Agent, focusing on thorough investigation of high-priority
targets before moving to others.
"""

import logging
import json
import time
from typing import List, Dict, Any, Optional, Set

# Configure logging
logger = logging.getLogger('research_strategy')

class ResearchStrategy:
    """
    Manages a prioritized research strategy with depth-first exploration.
    
    This class maintains a master list of investigation targets and implements
    logic for thorough, focused exploration instead of scattered investigation.
    """
    
    def __init__(self):
        """Initialize the research strategy."""
        self.master_todo_list = []  # List of investigation targets
        self.current_target = None
        self.completed_targets = []
        self.discovered_leads = []  # Leads found while investigating current target
        self.target_types = set()  # Types of targets investigated
        self.last_update_time = time.time()
        
    def add_target(self, target_info: Dict[str, Any]) -> None:
        """
        Add a target to the master todo list.
        
        Args:
            target_info: Information about the target including priority and rationale
        """
        # Ensure target has minimum required fields
        if not target_info.get('type'):
            logger.warning("Target missing required 'type' field, skipping")
            return
            
        # Add target type to our set of known types
        self.target_types.add(target_info.get('type'))
        
        # Ensure target has a priority
        if 'priority' not in target_info:
            target_info['priority'] = 5  # Default priority
        
        # Assign a unique ID to the target if it doesn't have one
        if 'id' not in target_info:
            target_id = self._generate_target_id(target_info)
            target_info['id'] = target_id
            
        # Check if target is already in the todo list (avoid duplicates)
        for existing in self.master_todo_list:
            if self._targets_are_equivalent(existing, target_info):
                logger.info(f"Skipping duplicate target: {target_info.get('type')} - " +
                           (f"{target_info.get('url')}" if target_info.get('type') in ['website', 'wayback', 'github'] else
                            f"{target_info.get('query')}" if target_info.get('type') == 'search' else "Unknown"))
                return
                
        # Check if target is already completed
        for completed in self.completed_targets:
            if self._targets_are_equivalent(completed, target_info):
                logger.info(f"Skipping already completed target: {target_info.get('type')} - " +
                           (f"{target_info.get('url')}" if target_info.get('type') in ['website', 'wayback', 'github'] else
                            f"{target_info.get('query')}" if target_info.get('type') == 'search' else "Unknown"))
                return
                
        # Add to master todo list
        self.master_todo_list.append(target_info)
        
        # Log the addition
        logger.info(f"Added target to todo list: {target_info.get('type')} - " +
                   (f"{target_info.get('url')}" if target_info.get('type') in ['website', 'wayback', 'github'] else
                    f"{target_info.get('query')}" if target_info.get('type') == 'search' else "Unknown") +
                   f" (Priority: {target_info.get('priority')})")
        
        # Update last update time
        self.last_update_time = time.time()
        
    def add_targets(self, targets: List[Dict[str, Any]]) -> None:
        """
        Add multiple targets to the master todo list.
        
        Args:
            targets: List of target information dictionaries
        """
        for target in targets:
            self.add_target(target)
            
    def get_next_target(self) -> Optional[Dict[str, Any]]:
        """
        Get the highest priority target from the todo list.
        
        Returns:
            The next target to investigate, or None if the list is empty
        """
        if not self.master_todo_list:
            return None
            
        # Sort by priority (descending)
        self.master_todo_list.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        # Get the highest priority target
        target = self.master_todo_list.pop(0)
        
        # Set as current target
        self.current_target = target
        
        # Log the selection
        logger.info(f"Selected target: {target.get('type')} - " +
                   (f"{target.get('url')}" if target.get('type') in ['website', 'wayback', 'github'] else
                    f"{target.get('query')}" if target.get('type') == 'search' else "Unknown") +
                   f" (Priority: {target.get('priority')})")
        
        return target
        
    def mark_target_complete(self, target: Dict[str, Any], discoveries: List[Dict[str, Any]] = None) -> None:
        """
        Mark a target as completed and move it to the completed list.
        
        Args:
            target: The target that was completed
            discoveries: Optional list of discoveries made during investigation
        """
        if target is None:
            return
            
        # Add completion timestamp
        target['completed_at'] = time.time()
        
        # Add discovery count if provided
        if discoveries is not None:
            target['discoveries_count'] = len(discoveries)
            
        # Move to completed list
        self.completed_targets.append(target)
        
        # Clear current target if it matches
        if self.current_target and self._targets_are_equivalent(self.current_target, target):
            self.current_target = None
            
        # Log the completion
        logger.info(f"Marked target as complete: {target.get('type')} - " +
                   (f"{target.get('url')}" if target.get('type') in ['website', 'wayback', 'github'] else
                    f"{target.get('query')}" if target.get('type') == 'search' else "Unknown"))
        
    def add_discovered_lead(self, lead: Dict[str, Any]) -> None:
        """
        Add a lead discovered during investigation but not immediately pursued.
        
        Args:
            lead: Information about the discovered lead
        """
        # Add timestamp when lead was discovered
        lead['discovered_at'] = time.time()
        
        # Add to discovered leads list
        self.discovered_leads.append(lead)
        
        # Log the discovery
        logger.info(f"Added discovered lead: {lead.get('type')} - " +
                   (f"{lead.get('url')}" if lead.get('type') in ['website', 'wayback', 'github'] else
                    f"{lead.get('query')}" if lead.get('type') == 'search' else "Unknown"))
    
    def update_priorities(self, priority_updates: Dict[str, Any] = None) -> None:
        """
        Update the priorities of targets in the todo list.
        
        Args:
            priority_updates: Optional dictionary of priority update rules
        """
        # If no specific updates provided, just re-sort the list
        if not priority_updates:
            self.master_todo_list.sort(key=lambda x: x.get('priority', 0), reverse=True)
            return
            
        # Apply priority updates to todo list
        for target in self.master_todo_list:
            target_type = target.get('type')
            
            # Apply type-specific updates
            if target_type in priority_updates.get('type_adjustments', {}):
                adjustment = priority_updates['type_adjustments'][target_type]
                target['priority'] = target.get('priority', 5) + adjustment
                
            # Apply domain-specific updates for website targets
            if target_type in ['website', 'wayback'] and 'domain_priorities' in priority_updates:
                domain = self._extract_domain(target.get('url', ''))
                if domain in priority_updates['domain_priorities']:
                    target['priority'] = max(target.get('priority', 5), 
                                            priority_updates['domain_priorities'][domain])
        
        # Re-sort the list
        self.master_todo_list.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        # Update last update time
        self.last_update_time = time.time()
        
        # Log the update
        logger.info("Updated target priorities")
        
    def process_discovered_leads(self) -> None:
        """
        Process discovered leads by adding them to the todo list.
        """
        if not self.discovered_leads:
            return
            
        # Add all discovered leads to the todo list
        leads_count = len(self.discovered_leads)
        for lead in self.discovered_leads:
            self.add_target(lead)
            
        # Clear the discovered leads list
        self.discovered_leads = []
        
        # Log the processing
        logger.info(f"Processed {leads_count} discovered leads")
        
    def merge_strategy_updates(self, strategy_updates: Dict[str, Any]) -> None:
        """
        Merge updates from a comprehensive strategy review.
        
        Args:
            strategy_updates: Dictionary with strategy updates from LLM
        """
        # Add new targets from the update
        if 'new_targets' in strategy_updates:
            self.add_targets(strategy_updates['new_targets'])
            
        # Update priorities
        if 'priority_updates' in strategy_updates:
            self.update_priorities(strategy_updates['priority_updates'])
            
        # Log the merge
        logger.info("Merged strategy updates from comprehensive review")
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the research strategy.
        
        Returns:
            Dictionary with status information
        """
        return {
            'todo_count': len(self.master_todo_list),
            'completed_count': len(self.completed_targets),
            'discovered_leads_count': len(self.discovered_leads),
            'target_types': list(self.target_types),
            'current_target': self.current_target,
        }
        
    def _generate_target_id(self, target: Dict[str, Any]) -> str:
        """Generate a unique ID for a target."""
        target_type = target.get('type', 'unknown')
        
        if target_type == 'website' and 'url' in target:
            return f"website:{target['url']}"
        elif target_type == 'wayback' and 'url' in target:
            return f"wayback:{target['url']}"
        elif target_type == 'search' and 'query' in target:
            return f"search:{target['query']}"
        elif target_type == 'github' and 'url' in target:
            return f"github:{target['url']}"
        else:
            # Fallback to using timestamp
            return f"{target_type}:{time.time()}"
            
    def _targets_are_equivalent(self, target1: Dict[str, Any], target2: Dict[str, Any]) -> bool:
        """
        Determine if two targets are equivalent (to avoid duplicates).
        
        Args:
            target1: First target
            target2: Second target
            
        Returns:
            True if the targets are equivalent, False otherwise
        """
        # Different types means different targets
        if target1.get('type') != target2.get('type'):
            return False
            
        target_type = target1.get('type')
        
        # Check equivalence based on type
        if target_type == 'website' and target1.get('url') == target2.get('url'):
            return True
        elif target_type == 'wayback' and target1.get('url') == target2.get('url'):
            return True
        elif target_type == 'search' and target1.get('query') == target2.get('query'):
            return True
        elif target_type == 'github' and target1.get('url') == target2.get('url'):
            return True
            
        # By default, not equivalent
        return False
        
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if not url:
            return ""
            
        # Simple domain extraction
        url = url.lower()
        url = url.replace("http://", "").replace("https://", "")
        domain = url.split("/")[0]
        
        return domain