#!/usr/bin/env python3
"""
Link extraction and crawling functionality for Narrahunt Phase 2.
"""

import re
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests

logger = logging.getLogger('narrahunt.crawl')

def extract_links(base_url, html_content):
    """
    Extract links from HTML content.
    
    Args:
        base_url: Base URL for resolving relative links
        html_content: HTML content to parse
        
    Returns:
        List of absolute URLs
    """
    links = []
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Check for base tag
        base_tag = soup.find('base')
        if base_tag and 'href' in base_tag.attrs:
            base_href = base_tag['href']
            # Update base_url if base tag is present
            base_url = urljoin(base_url, base_href)
        
        # Extract links from <a> tags
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            
            # Skip empty, javascript, and anchor links
            if (not href or 
                href.startswith('javascript:') or 
                href.startswith('#') or
                href.startswith('mailto:') or
                href.startswith('tel:')):
                continue
            
            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Validate URL scheme
            if urlparse(absolute_url).scheme in ('http', 'https'):
                links.append(absolute_url)
        
        logger.debug(f"Extracted {len(links)} links from {base_url}")
        return links
    
    except Exception as e:
        logger.error(f"Error extracting links from {base_url}: {e}")
        return []

def is_allowed_by_robots(url):
    """
    Check if a URL is allowed by the site's robots.txt.
    
    Args:
        url: URL to check
        
    Returns:
        Boolean indicating if crawling is allowed
    """
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        # Try to fetch robots.txt
        response = requests.get(robots_url, timeout=10)
        if response.status_code != 200:
            # No robots.txt or error fetching it - assume allowed
            return True
        
        # Check if our user agent is allowed
        user_agent = "NarrahuntBot"
        path = parsed.path
        
        # Very simple robots.txt parsing - a real implementation would be more thorough
        lines = response.text.splitlines()
        current_agent = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check for User-agent line
            if line.lower().startswith('user-agent:'):
                agent = line.split(':', 1)[1].strip()
                if agent == '*' or agent == user_agent:
                    current_agent = agent
                else:
                    current_agent = None
            
            # Check for Disallow line
            elif current_agent and line.lower().startswith('disallow:'):
                disallow_path = line.split(':', 1)[1].strip()
                if disallow_path and path.startswith(disallow_path):
                    return False
        
        # No matching disallow directive found
        return True
    
    except Exception as e:
        logger.warning(f"Error checking robots.txt for {url}: {e}")
        # Assume allowed on error
        return True