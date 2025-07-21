#!/usr/bin/env python3
"""
Main controller for Narrahunt Phase 2 recursive crawler.
Manages the crawl queue, fetches pages, and extracts Ethereum artifacts.
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from urllib.parse import urlparse

# Import crawler components
from url_queue import URLQueue
from fetch import fetch_page
from crawl import extract_links, is_allowed_by_robots
from artifact_extractor import extract_artifacts_from_html

# Set up logging
base_dir = '/home/computeruse/.anthropic/narrahunt_phase2'
os.makedirs(f'{base_dir}/results/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{base_dir}/results/logs/full_crawl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('narrahunt.main')

def load_config(config_path=f"{base_dir}/config.json"):
    """Load crawler configuration from file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        # Default configuration
        return {
            "crawl_delay": 2,
            "max_pages": 100,
            "max_depth": 3,
            "follow_redirects": True,
            "respect_robots": True
        }

def load_source_profiles(profiles_path=f"{base_dir}/source_profiles.json"):
    """Load source domain profiles."""
    try:
        with open(profiles_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading source profiles: {e}")
        # Default simple profile
        return {
            "ethereum.org": {
                "allowed": True,
                "seed_urls": ["https://ethereum.org/en/"]
            }
        }

def is_allowed_domain(url, profiles):
    """Check if a domain is allowed for crawling."""
    domain = urlparse(url).netloc
    
    # Check direct match
    if domain in profiles and profiles[domain].get("allowed", True):
        return True
    
    # Check pattern match
    for pattern, profile in profiles.items():
        if pattern.startswith("*.") and domain.endswith(pattern[2:]):
            return profile.get("allowed", True)
    
    # Default to disallowed
    return False

def run_crawler(test_mode=False):
    """Run the recursive crawler."""
    logger.info("Starting Narrahunt Phase 2 recursive crawler")
    
    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration: {config}")
    
    # Load source profiles
    profiles = load_source_profiles()
    logger.info(f"Loaded {len(profiles)} source profiles")
    
    # Initialize URL queue
    queue = URLQueue()
    
    # Check if we're resuming from a previous state
    if os.path.exists("queue_state.json") and not test_mode:
        queue.load_state()
        logger.info(f"Resumed queue from state file: {queue.pending_count()} pending, {queue.visited_count()} visited")
    
    # If queue is empty, initialize with seed URLs
    if queue.is_empty():
        for domain, profile in profiles.items():
            if "seed_urls" in profile:
                for url in profile["seed_urls"]:
                    queue.add_url(url, depth=0)
        logger.info(f"Initialized queue with {queue.pending_count()} seed URLs")
    
    # Crawling statistics
    stats = {
        "pages_fetched": 0,
        "pages_failed": 0,
        "artifacts_found": 0,
        "high_scoring_artifacts": 0,
        "start_time": time.time()
    }
    
    # Main crawl loop
    try:
        while not queue.is_empty():
            # Check if we've reached the maximum pages
            if stats["pages_fetched"] >= config.get("max_pages", 100) and not test_mode:
                logger.info(f"Reached maximum pages limit: {config.get('max_pages', 100)}")
                break
            
            # Get next URL from queue
            url, depth = queue.next_url()
            
            if not url:
                continue
            
            logger.info(f"Processing URL: {url} (depth: {depth})")
            
            # Check if domain is allowed
            if not is_allowed_domain(url, profiles):
                logger.info(f"Skipping disallowed domain: {url}")
                continue
            
            # Check robots.txt
            if config.get("respect_robots", True) and not is_allowed_by_robots(url):
                logger.info(f"Skipping URL disallowed by robots.txt: {url}")
                continue
            
            # Fetch page
            try:
                html_content, fetch_info = fetch_page(url)
                stats["pages_fetched"] += 1
                logger.info(f"Successfully fetched {url}")
                
                # Extract artifacts
                artifacts = extract_artifacts_from_html(
                    html_content, 
                    url=url, 
                    date=fetch_info.get("date")
                )
                
                stats["artifacts_found"] += len(artifacts)
                high_scoring = len([a for a in artifacts if a.get("score", 0) > 0])
                stats["high_scoring_artifacts"] += high_scoring
                
                if artifacts:
                    logger.info(f"Found {len(artifacts)} artifacts ({high_scoring} high-scoring) on {url}")
                
                # If we haven't reached max depth, extract and add links
                if depth < config.get("max_depth", 3):
                    links = extract_links(url, html_content)
                    
                    # Filter links by allowed domains
                    allowed_links = [link for link in links if is_allowed_domain(link, profiles)]
                    
                    # Add links to queue
                    for link in allowed_links:
                        queue.add_url(link, depth + 1)
                    
                    logger.info(f"Added {len(allowed_links)} new URLs to queue from {url}")
                
                # Save queue state periodically
                if stats["pages_fetched"] % 5 == 0:
                    queue.save_state()
                    logger.info(f"Saved queue state: {queue.pending_count()} pending, {queue.visited_count()} visited")
                
                # Respect crawl delay
                time.sleep(config.get("crawl_delay", 2))
                
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                stats["pages_failed"] += 1
                continue
    
    except KeyboardInterrupt:
        logger.info("Crawler stopped by user")
    
    # Save final queue state
    queue.save_state()
    
    # Calculate stats
    elapsed_time = time.time() - stats["start_time"]
    stats["elapsed_time"] = elapsed_time
    stats["pages_per_second"] = stats["pages_fetched"] / elapsed_time if elapsed_time > 0 else 0
    
    logger.info("Crawler finished")
    logger.info(f"Stats: {stats}")
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Narrahunt Phase 2 Crawler")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--reset", action="store_true", help="Reset queue state")
    
    args = parser.parse_args()
    
    # Reset queue state if requested
    if args.reset and os.path.exists("queue_state.json"):
        os.remove("queue_state.json")
        logger.info("Reset queue state")
    
    # Run crawler
    run_crawler(test_mode=args.test)