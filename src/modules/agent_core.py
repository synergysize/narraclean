"""
Agent Core - Core loop logic extracted from detective_agent.py

This module contains the main detective agent loop and core processing functions.
"""

import os
import logging
import json
import time
import sys
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urlparse, urljoin

# Configure logger
logger = logging.getLogger(__name__)

def start_investigation(agent):
    """Start the investigation process using focused, depth-first exploration."""
    logger.info("Starting investigation with focused exploration strategy...")
    
    # Initialize comprehensive research strategy
    initialize_research(agent)
    
    # Check if strategy has targets after initialization
    strategy_status = agent.research_strategy.get_status()
    if strategy_status['todo_count'] == 0:
        logger.error("Research strategy has no targets after initialization. Cannot proceed with investigation.")
        return []
        
    # Main investigation loop
    while should_continue_investigation(agent):
        agent.current_iteration += 1
        logger.info(f"\n{'='*80}\nStarting iteration {agent.current_iteration}/{agent.max_iterations}\n{'='*80}")
        
        # Check if we need to process any discovered leads
        if agent.current_iteration % 5 == 0:  # Every 5 iterations
            agent.research_strategy.process_discovered_leads()
            
        # Check if we need a strategy review
        if agent.current_iteration % 10 == 0:  # Every 10 iterations
            review_research_strategy(agent)
        
        # Get next investigation target using focused strategy
        target = agent.research_strategy.get_next_target()
        if not target:
            logger.warning("No more targets to investigate in strategy")
            
            # Perform comprehensive strategy review with LLM
            review_research_strategy(agent, force_update=True)
            
            # If still no targets after strategy review, exit
            if agent.research_strategy.get_status()['todo_count'] == 0:
                logger.info("No targets in research strategy after review. Ending investigation.")
                break
                
            continue
        
        # Set as current target
        agent.current_target = target
        
        # Execute the investigation thoroughly
        new_discoveries = execute_investigation(agent, target)
        
        # Track discoveries for this iteration
        agent.iteration_discoveries[agent.current_iteration] = new_discoveries
        
        # Mark current target as complete in strategy
        agent.research_strategy.mark_target_complete(target, new_discoveries)
        
        # Reset idle iterations if we found something
        if new_discoveries:
            agent.idle_iterations = 0
            
            # Instead of immediately consulting LLM, add these as leads to be investigated later
            for discovery in new_discoveries:
                # Check if discovery contains potential leads
                potential_leads = extract_leads_from_discovery(agent, discovery)
                for lead in potential_leads:
                    # Add to discovered leads, don't pursue immediately
                    agent.research_strategy.add_discovered_lead(lead)
            
            # Log the investigation results
            log_investigation_results(agent, target, new_discoveries, [])
        else:
            # No new discoveries in this iteration
            logger.info(f"No new discoveries in iteration {agent.current_iteration}, continuing with next target")
        
        # Save state after each iteration
        save_state(agent)
    
    logger.info(f"Investigation completed after {agent.current_iteration} iterations")
    logger.info(f"Total discoveries: {len(agent.discoveries)}")
    
    # Generate final report
    generate_investigation_report(agent)
    
    return agent.discoveries

def should_continue_investigation(agent):
    """Check if the investigation should continue."""
    # Check iteration count
    if agent.current_iteration >= agent.max_iterations:
        logger.info(f"Reached maximum iterations ({agent.max_iterations})")
        return False
        
    # Check idle iteration count
    if agent.idle_iterations >= agent.max_idle_iterations:
        logger.info(f"Reached maximum idle iterations ({agent.max_idle_iterations})")
        return False
        
    return True

def execute_investigation(agent, target: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Execute an investigation on a target, implementing proper depth-first exploration.
    
    This method ensures complete exploration of a domain before moving to the next one,
    by extracting all links from the current domain and adding them to the queue with
    the same priority as the current target.
    
    Args:
        agent: The detective agent instance
        target: The investigation target
        
    Returns:
        List of new discoveries
    """
    target_type = target.get('type', 'unknown')
    
    # Set current focus target for depth-first exploration
    if target_type in ['website', 'github', 'wayback']:
        domain = extract_domain(target.get('url', ''))
        if domain and not hasattr(agent, 'current_focus_target'):
            agent.current_focus_target = domain
            logger.info(f"Setting current focus target to: {agent.current_focus_target}")
        elif domain and agent.current_focus_target != domain:
            # Only switch focus if the current domain is exhausted
            if is_target_exhausted(agent, agent.current_focus_target):
                logger.info(f"Switching focus target from {agent.current_focus_target} to {domain}")
                agent.current_focus_target = domain
    
    # Track the target as partially explored until we've exhausted all related links
    target['exploration_status'] = 'partially_explored'
    
    # Execute the appropriate investigation method based on target type
    discoveries = []
    if target_type == 'website':
        discoveries = investigate_website(agent, target)
    elif target_type == 'search':
        discoveries = execute_search(agent, target)
    elif target_type == 'wayback':
        discoveries = investigate_wayback(agent, target)
    elif target_type == 'github':
        discoveries = investigate_github(agent, target)
    else:
        logger.warning(f"Unknown target type: {target_type}")
        return []
        
    # Check if this domain is now fully exhausted
    if target_type in ['website', 'github', 'wayback']:
        domain = extract_domain(target.get('url', ''))
        if domain and is_target_exhausted(agent, domain):
            logger.info(f"Domain {domain} is now fully explored")
            # Update all targets from this domain to mark them as fully explored
            for completed_target in agent.research_strategy.completed_targets:
                if completed_target.get('type') in ['website', 'github', 'wayback']:
                    if extract_domain(completed_target.get('url', '')) == domain:
                        completed_target['exploration_status'] = 'fully_explored'
            
            # If this was our current focus, we need to choose a new focus
            if agent.current_focus_target == domain:
                agent.current_focus_target = None
                
    return discoveries

def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        return urlparse(url).netloc
    except:
        return ""

def is_target_exhausted(agent, domain: str) -> bool:
    """
    Check if a domain has been fully explored.
    
    Args:
        agent: The detective agent instance
        domain: Domain to check
        
    Returns:
        Boolean indicating if domain is exhausted
    """
    # Check if there are any pending targets for this domain
    for target in agent.research_strategy.todo_targets:
        target_domain = extract_domain(target.get('url', ''))
        if target_domain == domain:
            return False
            
    return True

# Main function to initialize and run the detective agent
def main(objective=None, entity=None, max_iterations=50):
    """Main function to run the detective agent."""
    from src.detective_agent import DetectiveAgent
    
    if not objective:
        # If no objective specified, use command line arguments or default
        if len(sys.argv) > 1:
            objective = sys.argv[1]
        else:
            objective = "Investigate the early history of Ethereum and identify key contributors"
    
    if not entity:
        if len(sys.argv) > 2:
            entity = sys.argv[2]
        else:
            entity = "Ethereum"
            
    # Create and run agent
    agent = DetectiveAgent(objective, entity, max_iterations)
    agent.start_investigation()
    
    return 0