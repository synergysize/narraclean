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
from modules.url_queue import URLQueue
from modules.fetch import fetch_page
from modules.crawl import extract_links, is_allowed_by_robots
from modules.enhanced_artifact_detector import EnhancedArtifactDetector
from modules.utils import is_allowed_domain

# Set up logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

def load_config(config_path=f"{base_dir}/config/config.json"):
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
    run_crawler(test_mode=args.test)