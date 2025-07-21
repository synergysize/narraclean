from modules.utils import is_allowed_by_robots, is_allowed_domain

#!/usr/bin/env python3
"""
Enhanced Crawler Module for Narrahunt Phase 2.

This module provides a unified interface for crawling web content,
extracting links, and managing the crawl queue.
"""

import os
import time
import logging
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Set up logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_dir, 'logs', 'crawler.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('crawler')

# Import required components from local modules
from modules.fetch import fetch_page
from modules.crawl import extract_links, is_allowed_by_robots
from modules.url_queue import URLQueue
from modules.enhanced_artifact_detector import EnhancedArtifactDetector

class Crawler:
    """
    Enhanced crawler for web content with artifact extraction.
    """
    
    def __init__(self, config_path: Optional[str] = None, queue_state_path: Optional[str] = None):
        """
        Initialize the crawler.
        
        Args:
            config_path: Path to configuration file
            queue_state_path: Path to queue state file
        """
        self.config_path = config_path or os.path.join(base_dir, 'config', 'crawler_config.json')
        
        # Initialize URL queue
        self.queue = URLQueue(queue_state_path)
        
        # Load configuration
        self.config = self._load_config()
        
        # Statistics
        self.stats = {
            "pages_fetched": 0,
            "pages_failed": 0,
            "artifacts_found": 0,
            "high_scoring_artifacts": 0,
            "start_time": None,
            "end_time": None
        }
        
        logger.info("Crawler initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load crawler configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        # Default configuration
        default_config = {
            "crawl_delay": 2,
            "max_pages": 100,
            "max_depth": 3,
            "respect_robots": True,
            "allowed_domains": [
                # Removed hardcoded fallback domains - use LLM strategy only
                "medium.com",
                "twitter.com",
                "reddit.com",
                "duckduckgo.com",
                "archive.org",
                "web.archive.org"
            ]
        }
        
        # Save default config
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def add_urls(self, urls: List[str], depth: int = 0) -> int:
        """
        Add URLs to the crawl queue.
        
        Args:
            urls: List of URLs to add
            depth: Crawl depth for these URLs
            
        Returns:
            Number of URLs added
        """
        added_count = 0
        for url in urls:
            if self.queue.add_url(url, depth):
                added_count += 1
        
        logger.info(f"Added {added_count} URLs to the queue")
        return added_count
    
    def process_url(self, url: str, depth: int, extract_links_flag: bool = True) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Process a single URL, fetching content and extracting artifacts.
        
        Args:
            url: URL to process
            depth: Current crawl depth
            extract_links_flag: Whether to extract and follow links
            
        Returns:
            Tuple of (HTML content, list of artifacts)
        """
        logger.info(f"Processing URL: {url} (depth: {depth})")
        
        # Check robots.txt
        if self.config.get("respect_robots", True) and not is_allowed_by_robots(url):
            logger.info(f"Skipping URL disallowed by robots.txt: {url}")
            return None, []
        
        # Fetch content
        try:
            html_content, fetch_info = fetch_page(url)
            self.stats["pages_fetched"] += 1
            
            if not html_content:
                logger.warning(f"No content fetched from {url}")
                return None, []
            
            # Enhanced debug logging
            content_preview = html_content[:500] + "..." if len(html_content) > 500 else html_content
            logger.debug(f"Content preview from {url}: {content_preview}")
            
            # Extract artifacts using the enhanced detector
            logger.info(f"Extracting artifacts from {url}")
            # Initialize the detector once
            detector = EnhancedArtifactDetector()
            artifacts = detector.extract_artifacts(
                html_content, 
                url=url, 
                date=fetch_info.get("date"),
                objective=self.config.get("objective", "")
            )
            
            self.stats["artifacts_found"] += len(artifacts)
            high_scoring = len([a for a in artifacts if a.get("score", 0) > 0.7])
            self.stats["high_scoring_artifacts"] += high_scoring
            
            logger.info(f"Found {len(artifacts)} artifacts ({high_scoring} high-scoring) on {url}")
            
            # Extract links if needed
            if extract_links_flag and depth < self.config.get("max_depth", 3):
                links = extract_links(url, html_content)
                
                # Filter links by allowed domains
                allowed_links = []
                for link in links:
                    for domain in self.config.get("allowed_domains", []):
                        if domain in link:
                            allowed_links.append(link)
                            break
                
                # Add links to queue
                for link in allowed_links:
                    self.queue.add_url(link, depth + 1)
                
                logger.info(f"Added {len(allowed_links)} new URLs to queue from {url}")
            
            # Respect crawl delay
            time.sleep(self.config.get("crawl_delay", 2))
            
            return html_content, artifacts
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            self.stats["pages_failed"] += 1
            return None, []
    
    def crawl(self, max_pages: Optional[int] = None, extract_artifacts: bool = True) -> Dict[str, Any]:
        """
        Crawl URLs from the queue and extract artifacts.
        
        Args:
            max_pages: Maximum number of pages to crawl
            extract_artifacts: Whether to extract artifacts
            
        Returns:
            Crawl statistics
        """
        logger.info("Starting crawl")
        
        max_pages = max_pages or self.config.get("max_pages", 100)
        
        self.stats.update({
            "pages_fetched": 0,
            "pages_failed": 0,
            "artifacts_found": 0,
            "high_scoring_artifacts": 0,
            "start_time": datetime.now()
        })
        
        all_artifacts = []
        
        try:
            while not self.queue.is_empty() and self.stats["pages_fetched"] < max_pages:
                # Get next URL
                url, depth = self.queue.next_url()
                
                if not url:
                    continue
                
                # Process URL
                _, artifacts = self.process_url(url, depth, extract_links_flag=True)
                
                # Store artifacts
                if artifacts:
                    all_artifacts.extend(artifacts)
                
                # Save queue state periodically
                if self.stats["pages_fetched"] % 5 == 0:
                    self.queue.save_state()
                    logger.info(f"Queue state saved: {self.queue.pending_count()} pending, {self.queue.visited_count()} visited")
        
        except KeyboardInterrupt:
            logger.info("Crawl interrupted by user")
        
        # Update stats
        self.stats["end_time"] = datetime.now()
        elapsed_seconds = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        self.stats["elapsed_seconds"] = elapsed_seconds
        self.stats["pages_per_second"] = self.stats["pages_fetched"] / elapsed_seconds if elapsed_seconds > 0 else 0
        self.stats["all_artifacts"] = all_artifacts
        
        # Save final queue state
        self.queue.save_state()
        
        logger.info(f"Crawl completed. Processed {self.stats['pages_fetched']} pages, found {self.stats['artifacts_found']} artifacts")
        
        return self.stats
    
    def search(self, keywords: List[str], max_results: int = 10, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Search for content related to keywords.
        
        Args:
            keywords: List of keywords to search for
            max_results: Maximum number of results to return
            depth: Crawl depth
            
        Returns:
            List of search results
        """
        # Clear queue
        self.queue = URLQueue()
        
        # Create search URLs
        search_urls = []
        for keyword in keywords:
            # DuckDuckGo search
            search_urls.append(f"https://duckduckgo.com/html/?q={'+'.join(keyword.split())}")
            
            # GitHub search
            search_urls.append(f"https://github.com/search?q={'+'.join(keyword.split())}&type=code")
            
            # Removed hardcoded fallback - use LLM strategy only
        
        # Add URLs to queue
        self.add_urls(search_urls, depth=0)
        
        # Process URLs
        results = []
        processed_count = 0
        
        while not self.queue.is_empty() and processed_count < max_results * 3:  # Process 3x the number of URLs to find enough results
            url, depth = self.queue.next_url()
            
            if not url:
                continue
            
            html_content, artifacts = self.process_url(url, depth, extract_links_flag=(depth < 1))
            
            if html_content:
                result = {
                    "url": url,
                    "title": self._extract_title(html_content),
                    "content": self._extract_content(html_content),
                    "artifacts": artifacts
                }
                
                results.append(result)
            
            processed_count += 1
            
            # If we have enough results, stop
            if len(results) >= max_results:
                break
        
        logger.info(f"Search completed. Found {len(results)} results.")
        return results[:max_results]
    
    def _extract_title(self, html_content: str) -> str:
        """Extract page title from HTML content."""
        import re
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1).strip()
        return "Untitled"
    
    def _extract_content(self, html_content: str) -> str:
        """Extract main content from HTML."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style tags
            for tag in soup(['script', 'style']):
                tag.decompose()
            
            # Get text
            text = soup.get_text(separator=" ", strip=True)
            
            # Limit length
            if len(text) > 1000:
                text = text[:997] + "..."
            
            return text
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return html_content[:500] + "..." if len(html_content) > 500 else html_content

# Test the crawler
if __name__ == "__main__":
    print("Testing Enhanced Crawler")
    
    crawler = Crawler()
    
    # Test URLs - removed hardcoded fallback URLs
    test_urls = []
    
    crawler.add_urls(test_urls)
    
    # Crawl a few pages
    print("Crawling a few pages...")
    results = crawler.crawl(max_pages=3)
    
    print(f"Crawled {results['pages_fetched']} pages")
    print(f"Found {results['artifacts_found']} artifacts")
    print(f"High-scoring artifacts: {results['high_scoring_artifacts']}")
    
    # Test search
    print("\nTesting search...")
    search_results = crawler.search(["Vitalik Buterin ethereum"], max_results=2)
    
    for i, result in enumerate(search_results):
        print(f"\nResult {i+1}: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Artifacts: {len(result.get('artifacts', []))}")
    
    print("\nCrawler test complete")