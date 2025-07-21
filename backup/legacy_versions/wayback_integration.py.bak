#!/usr/bin/env python3
"""
Wayback Machine Integration for Narrahunt Phase 2.

This module provides integration with the Internet Archive's Wayback Machine
to fetch historical snapshots of websites for research purposes.
"""

import os
import re
import json
import time
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin

# Configure logging
base_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_dir, 'logs', 'wayback.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('wayback_integration')

class WaybackMachine:
    """
    Integrates with the Internet Archive's Wayback Machine to fetch historical snapshots.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the Wayback Machine integration.
        
        Args:
            cache_dir: Directory to cache wayback results
        """
        self.cache_dir = cache_dir or os.path.join(base_dir, 'cache', 'wayback')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # URL templates for Wayback API
        self.availability_url = "https://archive.org/wayback/available?url={url}&timestamp={timestamp}"
        self.cdx_url = "https://web.archive.org/cdx/search/cdx?url={url}&matchType=prefix&collapse=timestamp:4&limit=100&fl=original,timestamp,digest,mimetype,statuscode&from={from_date}&to={to_date}"
        
        # Cache for API responses
        self.response_cache = {}
    
    def _get_cache_file(self, url: str, timestamp: Optional[str] = None) -> str:
        """
        Get the cache file path for a URL and optional timestamp.
        
        Args:
            url: The URL to check
            timestamp: Optional timestamp (YYYYMMDD)
            
        Returns:
            Path to the cache file
        """
        domain = urlparse(url).netloc
        path = urlparse(url).path.replace('/', '_')
        if timestamp:
            return os.path.join(self.cache_dir, f"{domain}{path}_{timestamp}.json")
        else:
            return os.path.join(self.cache_dir, f"{domain}{path}.json")
    
    def check_availability(self, url: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if a URL is available in the Wayback Machine.
        
        Args:
            url: The URL to check
            timestamp: Optional timestamp (YYYYMMDD)
            
        Returns:
            Dictionary with availability information
        """
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        
        # Check cache first
        cache_key = f"{url}_{timestamp}" if timestamp else url
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        cache_file = self._get_cache_file(url, timestamp)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    result = json.load(f)
                self.response_cache[cache_key] = result
                return result
            except:
                logger.warning(f"Failed to load cache file: {cache_file}")
        
        # Prepare API URL
        timestamp_param = timestamp or ""
        api_url = self.availability_url.format(url=url, timestamp=timestamp_param)
        
        # Make request
        try:
            logger.info(f"Checking Wayback availability for {url}")
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Cache result
                    with open(cache_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    self.response_cache[cache_key] = result
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing Wayback JSON response: {e}")
                    return {"archived_snapshots": {}}
            else:
                logger.error(f"Wayback API error: {response.status_code}")
                return {"archived_snapshots": {}}
        
        except Exception as e:
            logger.error(f"Error checking Wayback availability: {e}")
            return {"archived_snapshots": {}}
    
    def get_snapshots(self, url: str, from_date: str = "2013", to_date: str = "2017") -> List[Dict[str, Any]]:
        """
        Get a list of snapshots for a URL in a given date range.
        
        Args:
            url: The URL to check
            from_date: Start date (YYYY or YYYYMMDD)
            to_date: End date (YYYY or YYYYMMDD)
            
        Returns:
            List of snapshot information dictionaries
        """
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        
        # Normalize dates
        if len(from_date) == 4:
            from_date = f"{from_date}0101"
        if len(to_date) == 4:
            to_date = f"{to_date}1231"
        
        # Check cache first
        cache_key = f"{url}_{from_date}_{to_date}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        cache_file = os.path.join(self.cache_dir, f"{urlparse(url).netloc}_{from_date}_{to_date}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    result = json.load(f)
                self.response_cache[cache_key] = result
                return result
            except:
                logger.warning(f"Failed to load cache file: {cache_file}")
        
        # Prepare API URL
        api_url = self.cdx_url.format(url=url, from_date=from_date, to_date=to_date)
        
        # Make request
        try:
            logger.info(f"Getting Wayback snapshots for {url} from {from_date} to {to_date}")
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                # Parse CDX response (space-separated values)
                snapshots = []
                for line in response.text.splitlines():
                    if line.strip():
                        parts = line.split(' ')
                        if len(parts) >= 5:
                            snapshot = {
                                "original": parts[0],
                                "timestamp": parts[1],
                                "digest": parts[2],
                                "mimetype": parts[3],
                                "statuscode": parts[4],
                                "wayback_url": f"https://web.archive.org/web/{parts[1]}/{parts[0]}"
                            }
                            snapshots.append(snapshot)
                
                # Cache result
                with open(cache_file, 'w') as f:
                    json.dump(snapshots, f, indent=2)
                
                self.response_cache[cache_key] = snapshots
                return snapshots
            else:
                logger.error(f"Wayback CDX API error: {response.status_code}")
                return []
        
        except Exception as e:
            logger.error(f"Error getting Wayback snapshots: {e}")
            return []
    
    def get_wayback_url(self, url: str, timestamp: Optional[str] = None) -> Optional[str]:
        """
        Get the Wayback Machine URL for a specific snapshot.
        
        Args:
            url: The original URL
            timestamp: Optional timestamp (YYYYMMDD)
            
        Returns:
            Wayback Machine URL or None if not available
        """
        availability = self.check_availability(url, timestamp)
        
        if "archived_snapshots" in availability and "closest" in availability["archived_snapshots"]:
            snapshot = availability["archived_snapshots"]["closest"]
            return snapshot.get("url")
        
        return None
    
    def get_crawlable_snapshots(self, url: str, from_year: int = 2013, to_year: int = 2017,
                              limit_per_year: int = 4) -> List[Dict[str, Any]]:
        """
        Get a list of crawlable snapshots for a URL, with even distribution across years.
        
        Args:
            url: The URL to check
            from_year: Start year
            to_year: End year
            limit_per_year: Maximum snapshots per year
            
        Returns:
            List of snapshot information dictionaries with wayback URLs
        """
        crawlable_snapshots = []
        
        # Get snapshots for each year
        for year in range(from_year, to_year + 1):
            from_date = f"{year}0101"
            to_date = f"{year}1231"
            
            snapshots = self.get_snapshots(url, from_date, to_date)
            
            # Filter to HTML content
            html_snapshots = [s for s in snapshots if s.get("mimetype", "").startswith("text/html") and s.get("statuscode") == "200"]
            
            # Evenly distribute through the year
            if html_snapshots:
                if len(html_snapshots) <= limit_per_year:
                    selected = html_snapshots
                else:
                    # Select snapshots evenly distributed throughout the year
                    step = len(html_snapshots) // limit_per_year
                    selected = [html_snapshots[i] for i in range(0, len(html_snapshots), step)][:limit_per_year]
                
                crawlable_snapshots.extend(selected)
        
        return crawlable_snapshots
    
    def generate_wayback_urls_for_target(self, url: str, objective: str) -> List[str]:
        """
        Generate a list of Wayback Machine URLs for a target based on the objective.
        
        Args:
            url: The target URL
            objective: The research objective
            
        Returns:
            List of Wayback Machine URLs to crawl
        """
        # Determine date range based on objective
        if "name" in objective.lower() and "vitalik" in objective.lower():
            # For name artifacts around Vitalik, focus on early years
            from_year = 2013
            to_year = 2016
        elif "wallet" in objective.lower():
            # For wallet artifacts, include more recent years
            from_year = 2014
            to_year = 2018
        elif "code" in objective.lower():
            # For code artifacts, focus on development years
            from_year = 2013
            to_year = 2017
        else:
            # Default range
            from_year = 2013
            to_year = 2017
        
        # Get crawlable snapshots
        snapshots = self.get_crawlable_snapshots(url, from_year, to_year)
        
        # Extract wayback URLs
        wayback_urls = [s["wayback_url"] for s in snapshots]
        
        logger.info(f"Generated {len(wayback_urls)} Wayback URLs for {url}")
        return wayback_urls
    
    def enrich_url_list_with_wayback(self, urls: List[str], objective: str) -> List[str]:
        """
        Enrich a list of URLs with their Wayback Machine snapshots.
        
        Args:
            urls: List of original URLs
            objective: The research objective
            
        Returns:
            Original URLs plus relevant Wayback Machine URLs
        """
        enriched_urls = urls.copy()
        wayback_urls = []
        
        for url in urls:
            # Skip URLs that are already Wayback URLs
            if "web.archive.org" in url:
                continue
            
            # Removed hardcoded fallback - use LLM strategy only
            # Process all URLs from the strategy, don't filter based on hardcoded domains
            if True:  # Always process URLs from the strategy
                # Get wayback URLs for this target
                target_wayback_urls = self.generate_wayback_urls_for_target(url, objective)
                wayback_urls.extend(target_wayback_urls)
        
        # Add unique wayback URLs to the list
        for url in wayback_urls:
            if url not in enriched_urls:
                enriched_urls.append(url)
        
        logger.info(f"Enriched URL list with {len(wayback_urls)} Wayback URLs")
        return enriched_urls

if __name__ == "__main__":
    # Test the Wayback Machine integration
    wayback = WaybackMachine()
    
    test_url = "vitalik.ca"
    test_objective = "Find name artifacts around Vitalik Buterin"
    
    print(f"Testing Wayback Machine integration with {test_url}")
    
    # Check availability
    availability = wayback.check_availability(test_url)
    if "archived_snapshots" in availability and "closest" in availability["archived_snapshots"]:
        print(f"URL is available in Wayback Machine: {availability['archived_snapshots']['closest'].get('url')}")
    else:
        print("URL is not available in Wayback Machine")
    
    # Get snapshots
    snapshots = wayback.get_snapshots(test_url, "2013", "2015")
    print(f"Found {len(snapshots)} snapshots from 2013-2015")
    
    # Get crawlable snapshots
    crawlable = wayback.get_crawlable_snapshots(test_url, 2013, 2015)
    print(f"Generated {len(crawlable)} crawlable snapshots")
    
    # Test URL enrichment
    test_urls = [
        "https://vitalik.ca",
        "https://github.com/vbuterin",
        "https://ethereum.org"
    ]
    
    enriched = wayback.enrich_url_list_with_wayback(test_urls, test_objective)
    print(f"Enriched {len(test_urls)} URLs to {len(enriched)} URLs with Wayback Machine integration")