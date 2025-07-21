#!/usr/bin/env python3
"""
Page fetcher for Narrahunt Phase 2.
Handles fetching web pages with error handling, retries, and timeouts.
"""

import requests
import time
import logging
from urllib.parse import urlparse
from datetime import datetime

logger = logging.getLogger('narrahunt.fetch')

# Default headers
DEFAULT_HEADERS = {
    "User-Agent": "NarrahuntBot/2.0 (https://narrahunt.org/bot.html; bot@narrahunt.org)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}

def fetch_page(url, max_retries=3, timeout=30):
    """
    Fetch a web page with retries and timeouts.
    
    Args:
        url: URL to fetch
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (content, info_dict)
    """
    logger.info(f"Fetching {url}")
    
    headers = DEFAULT_HEADERS.copy()
    
    # Track attempt number
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Attempt {attempt}/{max_retries} for {url}")
            
            # Make the request
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True
            )
            
            # Check if successful
            if response.status_code == 200:
                # Validate response has content
                if not hasattr(response, 'text') or response.text is None:
                    logger.warning(f"Response from {url} has no text content")
                    return None, {"error": "No text content in response"}
                
                # Check if content is HTML
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' in content_type or 'application/xhtml+xml' in content_type:
                    # Extract some metadata
                    info = {
                        "url": response.url,  # Final URL after redirects
                        "status_code": response.status_code,
                        "content_type": content_type,
                        "date": response.headers.get('Date', datetime.now().isoformat()),
                        "fetch_time": time.time()
                    }
                    
                    logger.info(f"Successfully fetched {url} ({len(response.text)} bytes)")
                    return response.text, info
                else:
                    logger.warning(f"Skipping non-HTML content for {url}: {content_type}")
                    return None, {"error": "Not HTML content"}
            
            # Handle 404 errors immediately without retry
            elif response.status_code == 404:
                logger.warning(f"Page not found (404) for {url}")
                return None, {"error": "Page not found (404)"}
            
            # Handle other errors with backoff
            else:
                logger.warning(f"HTTP error {response.status_code} for {url}")
                
                # If we have more retries, wait and try again
                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.debug(f"Waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                else:
                    return None, {"error": f"HTTP error {response.status_code}"}
        
        except requests.RequestException as e:
            logger.warning(f"Request error for {url}: {e}")
            
            # If we have more retries, wait and try again
            if attempt < max_retries:
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.debug(f"Waiting {wait_time}s before retry")
                time.sleep(wait_time)
            else:
                return None, {"error": str(e)}
    
    return None, {"error": f"Failed after {max_retries} attempts"}

def is_allowed_by_robots(url):
    """
    Check if a URL is allowed by the site's robots.txt.
    This is a simple implementation, a real one would parse the robots.txt file.
    
    Args:
        url: URL to check
        
    Returns:
        Boolean indicating if crawling is allowed
    """
    # This is a simplified check - would normally parse robots.txt
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # In a real implementation, would fetch and parse robots.txt
    # For now, allow everything
    return True