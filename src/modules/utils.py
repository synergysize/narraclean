# Common extracted utilities
import requests
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

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

def is_allowed_domain(url: str, allowed_domains=None) -> bool:
    """
    Check if a domain is allowed for crawling.
    
    Args:
        url: URL to check
        allowed_domains: List of allowed domains
        
    Returns:
        Boolean indicating if domain is allowed
    """
    if not allowed_domains:
        allowed_domains = []
        
    domain = urlparse(url).netloc
    
    # Allow any subdomain of allowed domains
    for allowed_domain in allowed_domains:
        if domain == allowed_domain or domain.endswith(f".{allowed_domain}"):
            return True
    
    return False
    

