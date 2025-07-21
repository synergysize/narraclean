"""
Routing - Objective logic and orchestration

This module contains functions for managing objectives, routing tasks,
and orchestrating the research process.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

# Configure logger
logger = logging.getLogger(__name__)

def review_research_strategy(agent, force_update: bool = False):
    """
    Perform a comprehensive review of the research strategy.
    
    Args:
        agent: The detective agent instance
        force_update: If True, force a strategy update even if not scheduled
    """
    # Skip if not forced and we've recently updated
    if not force_update and time.time() - agent.research_strategy.last_update_time < 600:  # 10 minutes
        return
        
    logger.info("Performing comprehensive research strategy review...")
    
    # Get current strategy status
    strategy_status = agent.research_strategy.get_status()
    
    # Prepare discoveries context
    recent_discoveries = []
    for iteration in range(max(0, agent.current_iteration - 10), agent.current_iteration + 1):
        if iteration in agent.iteration_discoveries:
            recent_discoveries.extend(agent.iteration_discoveries[iteration])
            
    # Limit to most recent discoveries
    recent_discoveries = recent_discoveries[-10:]
    
    # Build context for the LLM
    context = {
        'objective': agent.objective,
        'entity': agent.entity,
        'current_iteration': agent.current_iteration,
        'total_discoveries': len(agent.discoveries),
        'recent_discoveries': recent_discoveries,
        'current_strategy': strategy_status,
        'completed_targets': agent.research_strategy.completed_targets[-10:],  # Last 10 completed targets
        'todo_count': strategy_status['todo_count']
    }
    
    # Get strategy recommendations from LLM
    strategy_updates = get_strategy_recommendations(agent, context)
    
    # Apply the updates to our strategy
    if strategy_updates:
        agent.research_strategy.merge_strategy_updates(strategy_updates)
        
    logger.info(f"Research strategy updated. New todo count: {agent.research_strategy.get_status()['todo_count']}")

def get_strategy_recommendations(agent, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get strategy recommendations from LLM.
    
    Args:
        agent: The detective agent instance
        context: Context information for the LLM
        
    Returns:
        Dictionary with strategy updates
    """
    # Construct a detailed prompt
    discoveries_text = format_discoveries_for_prompt(agent, context.get('recent_discoveries', []))
    completed_targets_text = format_targets_for_prompt(agent, context.get('completed_targets', []))
    
    prompt = f"""
You are a research strategist helping to investigate: "{context['objective']}"

Current status:
- Iteration: {context['current_iteration']}
- Total discoveries: {context['total_discoveries']}
- Remaining tasks: {context['todo_count']}

Recent discoveries:
{discoveries_text}

Recently completed targets:
{completed_targets_text}

Based on this information, provide updated research priorities and specific targets to investigate.
Focus on the most promising leads that will help achieve the research objective.

Please respond with a JSON object containing:
1. A list of specific URLs to investigate (with type and priority)
2. Search queries that should be performed
3. Any specific Wayback Machine targets that should be checked

Example format:
{{
  "website_targets": [
    {{"url": "https://example.com/relevant-page", "priority": 5}}
  ],
  "search_targets": [
    {{"query": "specific search query", "priority": 3}}
  ],
  "wayback_targets": [
    {{"url": "https://example.com", "year": 2015, "priority": 4}}
  ],
  "github_targets": [
    {{"url": "https://github.com/username/repo", "priority": 4}}
  ]
}}
"""
    
    # Try to get a response from the LLM
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Get LLM instance
            llm = agent._get_llm_instance()
            
            # Get response
            response = llm.run_prompt(prompt, max_tokens=1500)
            
            # Extract JSON from response
            strategy_updates = extract_json_from_text(response)
            
            if strategy_updates:
                return strategy_updates
                
            logger.warning(f"Failed to extract valid JSON from LLM response (attempt {attempt+1}/{max_attempts})")
        except Exception as e:
            logger.error(f"Error getting strategy recommendations: {e}")
            
        if attempt < max_attempts - 1:
            time.sleep(2)  # Wait before retrying
    
    # If we get here, all attempts failed
    logger.error("Failed to get strategy recommendations after multiple attempts")
    return {}

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text."""
    try:
        # Try to find JSON in the text
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
    except:
        pass
        
    return {}

def format_discoveries_for_prompt(agent, discoveries: List[Dict[str, Any]]) -> str:
    """Format discoveries for inclusion in prompts."""
    if not discoveries:
        return "No recent discoveries."
    
    formatted = []
    for d in discoveries:
        content = d.get('content', '')
        if len(content) > 200:
            content = content[:197] + "..."
        formatted.append(f"- {d.get('title', 'Untitled')}: {content}")
    
    return "\n".join(formatted)

def format_targets_for_prompt(agent, targets: List[Dict[str, Any]]) -> str:
    """Format targets for inclusion in prompts."""
    if not targets:
        return "No recent targets."
    
    formatted = []
    for t in targets:
        target_type = t.get('type', 'unknown')
        if target_type == 'website':
            formatted.append(f"- Website: {t.get('url', '')}")
        elif target_type == 'search':
            formatted.append(f"- Search: {t.get('query', '')}")
        elif target_type == 'wayback':
            formatted.append(f"- Wayback: {t.get('url', '')} ({t.get('year', 'unknown')})")
        elif target_type == 'github':
            formatted.append(f"- GitHub: {t.get('url', '')}")
    
    return "\n".join(formatted)

def extract_leads_from_discovery(agent, discovery: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract potential leads from a discovery.
    
    Args:
        agent: The detective agent instance
        discovery: The discovery to extract leads from
        
    Returns:
        List of potential leads
    """
    leads = []
    
    # Check if the discovery has content
    if 'content' in discovery:
        # Extract URLs from content
        urls = agent._extract_urls_from_text(discovery['content'])
        
        # Filter out irrelevant URLs
        relevant_urls = [url for url in urls if agent._is_url_relevant(url)]
        
        # Add each URL as a potential lead
        for url in relevant_urls:
            # Determine target type
            if 'github.com' in url:
                target_type = 'github'
            elif 'web.archive.org' in url:
                target_type = 'wayback'
            else:
                target_type = 'website'
                
            # Check if we should use Wayback Machine for this URL
            if target_type == 'website' and agent._should_check_wayback(url):
                # Also add a Wayback version of this lead
                parsed_url = urlparse(url)
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                leads.append({
                    'type': 'wayback',
                    'url': base_url,
                    'year': 'all',  # Check all years
                    'priority': 3,  # Medium priority
                    'source': discovery.get('url', 'unknown')
                })
            
            # Add the original URL as a lead
            leads.append({
                'type': target_type,
                'url': url,
                'priority': 4,  # Medium-high priority since it came from a discovery
                'source': discovery.get('url', 'unknown')
            })
    
    return leads