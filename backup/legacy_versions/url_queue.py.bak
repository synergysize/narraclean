#!/usr/bin/env python3
"""
URL queue manager for Narrahunt Phase 2.
Handles storage and retrieval of URLs to crawl, with deduplication and state persistence.
"""

import os
import json
import time
from urllib.parse import urlparse, urlunparse
import hashlib
import logging
import tempfile
import shutil

logger = logging.getLogger('narrahunt.queue')

class URLQueue:
    """Manages the crawler's URL queue with state persistence."""
    
    def __init__(self, state_file=None):
        """Initialize the URL queue."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.state_file = state_file or os.path.join(base_dir, "queue_state.json")
        self.pending = {}  # url_hash -> {"url": url, "depth": depth, "added": timestamp}
        self.visited = {}  # url_hash -> {"url": url, "visited": timestamp}
        self.domains_last_visit = {}  # domain -> timestamp
    
    def add_url(self, url, depth=0):
        """Add a URL to the queue if not already visited or pending."""
        # Normalize the URL
        url = self._normalize_url(url)
        if not url:
            return False
        
        # Generate URL hash
        url_hash = self._hash_url(url)
        
        # Check if already visited or pending
        if url_hash in self.visited or url_hash in self.pending:
            return False
        
        # Add to pending
        self.pending[url_hash] = {
            "url": url,
            "depth": depth,
            "added": time.time()
        }
        
        return True
    
    def next_url(self):
        """Get the next URL to process, respecting domain-based rate limiting."""
        if not self.pending:
            return None, None
        
        current_time = time.time()
        min_delay = 1.0  # Minimum delay between requests to same domain
        
        # Find a URL from a domain we haven't visited recently
        for url_hash, info in list(self.pending.items()):
            url = info["url"]
            domain = self._get_domain(url)
            
            # Check if we've visited this domain recently
            last_visit = self.domains_last_visit.get(domain, 0)
            if current_time - last_visit < min_delay:
                continue
            
            # Update domain visit time
            self.domains_last_visit[domain] = current_time
            
            # Move from pending to visited
            self.visited[url_hash] = {
                "url": url,
                "visited": current_time
            }
            
            # Remove from pending
            del self.pending[url_hash]
            
            return url, info["depth"]
        
        # If all domains are rate-limited, wait and return the first URL
        if self.pending:
            url_hash = next(iter(self.pending))
            info = self.pending[url_hash]
            url = info["url"]
            domain = self._get_domain(url)
            
            # Update domain visit time
            self.domains_last_visit[domain] = current_time
            
            # Move from pending to visited
            self.visited[url_hash] = {
                "url": url,
                "visited": current_time
            }
            
            # Remove from pending
            del self.pending[url_hash]
            
            return url, info["depth"]
        
        return None, None
    
    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.pending) == 0
    
    def pending_count(self):
        """Get the number of pending URLs."""
        return len(self.pending)
    
    def visited_count(self):
        """Get the number of visited URLs."""
        return len(self.visited)
    
    def save_state(self):
        """Save the current queue state to file."""
        state = {
            "pending": self.pending,
            "visited": self.visited,
            "domains_last_visit": self.domains_last_visit
        }
        
        # Define temp_file outside the try block so it's available in the except block
        temp_file = self.state_file + '.tmp'
        
        try:
            # Create backup of existing file
            if os.path.exists(self.state_file):
                backup_file = self.state_file + '.backup'
                shutil.copy2(self.state_file, backup_file)
            
            # Write to temporary file first
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Atomic move
            shutil.move(temp_file, self.state_file)
            logger.debug(f"Saved queue state: {len(self.pending)} pending, {len(self.visited)} visited")
            return True
        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                os.remove(temp_file)
            logger.error(f"Error saving queue state: {e}")
            return False
    
    def load_state(self):
        """Load queue state from file, trying backup if main file is corrupted."""
        if not os.path.exists(self.state_file):
            # Check for backup
            backup_file = self.state_file + '.backup'
            if os.path.exists(backup_file):
                logger.info(f"Main state file not found, but backup exists. Loading from backup.")
                try:
                    with open(backup_file, 'r') as f:
                        state = json.load(f)
                    
                    self.pending = state.get("pending", {})
                    self.visited = state.get("visited", {})
                    self.domains_last_visit = state.get("domains_last_visit", {})
                    
                    logger.info(f"Loaded queue state from backup: {len(self.pending)} pending, {len(self.visited)} visited")
                    
                    # Restore the backup as main file
                    shutil.copy2(backup_file, self.state_file)
                    logger.info(f"Restored backup file to main state file")
                    
                    return True
                except Exception as e:
                    logger.error(f"Error loading queue state from backup: {e}")
                    return False
            else:
                logger.info(f"No queue state file found at {self.state_file}")
                return False
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.pending = state.get("pending", {})
            self.visited = state.get("visited", {})
            self.domains_last_visit = state.get("domains_last_visit", {})
            
            logger.info(f"Loaded queue state: {len(self.pending)} pending, {len(self.visited)} visited")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing queue state file: {e}")
            
            # Try loading from backup
            backup_file = self.state_file + '.backup'
            if os.path.exists(backup_file):
                logger.info(f"Attempting to load from backup file after main file corruption")
                try:
                    with open(backup_file, 'r') as f:
                        state = json.load(f)
                    
                    self.pending = state.get("pending", {})
                    self.visited = state.get("visited", {})
                    self.domains_last_visit = state.get("domains_last_visit", {})
                    
                    logger.info(f"Loaded queue state from backup: {len(self.pending)} pending, {len(self.visited)} visited")
                    
                    # Restore the backup as main file
                    shutil.copy2(backup_file, self.state_file)
                    logger.info(f"Restored backup file to main state file")
                    
                    return True
                except Exception as nested_e:
                    logger.error(f"Error loading queue state from backup: {nested_e}")
            else:
                logger.warning("No backup file available to recover from corrupted state")
            
            return False
        except Exception as e:
            logger.error(f"Error loading queue state: {e}")
            return False
    
    def _normalize_url(self, url):
        """Normalize a URL by removing fragments and query parameters."""
        try:
            parsed = urlparse(url)
            
            # Ensure URL has a scheme
            if not parsed.scheme:
                url = "https://" + url
                parsed = urlparse(url)
            
            # Skip non-HTTP URLs
            if parsed.scheme not in ('http', 'https'):
                return None
            
            # Normalize by removing fragment
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc.lower(),
                parsed.path,
                parsed.params,
                parsed.query,
                ''  # No fragment
            ))
            
            return normalized
        except Exception:
            return None
    
    def _hash_url(self, url):
        """Generate a hash for a URL."""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def _get_domain(self, url):
        """Extract domain from URL."""
        return urlparse(url).netloc.lower()