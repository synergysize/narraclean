"""
Extractors - Artifact and name extraction functions

This module contains functions for extracting artifacts, names, and other
relevant information from content found during the detective process.
"""

import re
import json
import hashlib
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Configure logger
logger = logging.getLogger(__name__)

def extract_artifacts_from_html(html_content, url="", date=None):
    """
    Extract artifacts from HTML content.
    
    Args:
        html_content: HTML content to parse
        url: Source URL (optional)
        date: Date of the content (optional)
        
    Returns:
        List of extracted artifacts
    """
    if not html_content:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Track unique artifacts by hash
        artifact_hashes = set()
        
        # Extract artifacts
        artifacts = []
        
        # Extract Solidity contracts
        artifacts.extend(extract_solidity_contracts(soup, url, date, artifact_hashes))
        
        # Extract wallet addresses
        artifacts.extend(extract_wallet_addresses(soup, url, date, artifact_hashes))
        
        # Extract private keys
        artifacts.extend(extract_private_keys(soup, url, date, artifact_hashes))
        
        # Extract JSON keystores
        artifacts.extend(extract_json_keystores(soup, url, date, artifact_hashes))
        
        # Extract seed phrases
        artifacts.extend(extract_seed_phrases(soup, url, date, artifact_hashes))
        
        # Extract API keys
        artifacts.extend(extract_api_keys(soup, url, date, artifact_hashes))
        
        return artifacts
    except Exception as e:
        logger.error(f"Error extracting artifacts: {e}")
        return []

def extract_solidity_contracts(soup, url, date, artifact_hashes):
    """Extract Solidity smart contracts from HTML."""
    artifacts = []
    
    # Regular expression for Solidity contract definition
    contract_pattern = re.compile(r'contract\s+(\w+)\s*{.*?}', re.DOTALL)
    
    # Look in <pre>, <code>, and text nodes for Solidity code
    for element in soup.find_all(['pre', 'code']):
        text = element.get_text()
        
        # Look for Solidity contracts
        for match in contract_pattern.finditer(text):
            contract_text = match.group(0)
            contract_name = match.group(1)
            
            # Generate hash to avoid duplicates
            artifact_hash = generate_hash(contract_text)
            
            if artifact_hash in artifact_hashes:
                continue
                
            artifact_hashes.add(artifact_hash)
            
            # Score the artifact
            score = score_artifact(url, contract_text, date)
            
            # Create artifact object
            artifact = {
                'type': 'solidity_contract',
                'name': contract_name,
                'content': contract_text,
                'url': url,
                'date': date.isoformat() if date else None,
                'score': score,
                'hash': artifact_hash,
                'location': find_location(soup, contract_text)
            }
            
            artifacts.append(artifact)
    
    return artifacts

def extract_wallet_addresses(soup, url, date, artifact_hashes):
    """Extract cryptocurrency wallet addresses from HTML."""
    artifacts = []
    
    # Define patterns for different wallet address types
    patterns = {
        'ethereum': re.compile(r'0x[a-fA-F0-9]{40}'),
        'bitcoin': re.compile(r'(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}'),
    }
    
    # Extract text from the soup
    text = soup.get_text()
    
    # Extract addresses of each type
    for wallet_type, pattern in patterns.items():
        for match in pattern.finditer(text):
            address = match.group(0)
            
            # Generate hash to avoid duplicates
            artifact_hash = generate_hash(address)
            
            if artifact_hash in artifact_hashes:
                continue
                
            artifact_hashes.add(artifact_hash)
            
            # Score the artifact
            score = score_artifact(url, address, date)
            
            # Create artifact object
            artifact = {
                'type': f'{wallet_type}_address',
                'content': address,
                'url': url,
                'date': date.isoformat() if date else None,
                'score': score,
                'hash': artifact_hash,
                'location': find_location(soup, address)
            }
            
            artifacts.append(artifact)
    
    return artifacts

def extract_private_keys(soup, url, date, artifact_hashes):
    """Extract private keys from HTML."""
    artifacts = []
    
    # Define patterns for different private key formats
    patterns = {
        'ethereum_private_key': re.compile(r'0x[a-fA-F0-9]{64}'),
        'mnemonic': re.compile(r'[a-z]{3,15}( [a-z]{3,15}){11,23}')
    }
    
    # Extract text from the soup
    text = soup.get_text()
    
    # Extract private keys of each type
    for key_type, pattern in patterns.items():
        for match in pattern.finditer(text):
            key = match.group(0)
            
            # Generate hash to avoid duplicates
            artifact_hash = generate_hash(key)
            
            if artifact_hash in artifact_hashes:
                continue
                
            artifact_hashes.add(artifact_hash)
            
            # Score the artifact
            score = score_artifact(url, key, date)
            
            # Create artifact object
            artifact = {
                'type': key_type,
                'content': key,
                'url': url,
                'date': date.isoformat() if date else None,
                'score': score,
                'hash': artifact_hash,
                'location': find_location(soup, key)
            }
            
            artifacts.append(artifact)
    
    return artifacts

def score_artifact(url, content, date):
    """Score the relevance and significance of an artifact."""
    score = 5.0  # Default mid-level score
    
    # Adjust score based on content
    if "private" in content.lower() or "secret" in content.lower():
        score += 2.0
        
    if "key" in content.lower() or "password" in content.lower():
        score += 1.5
        
    if "test" in url.lower() or "example" in url.lower():
        score -= 2.0
        
    # Score based on recency (if date provided)
    if date:
        try:
            artifact_date = date
            current_date = datetime.now()
            
            # Calculate age in days
            age_days = (current_date - artifact_date).days
            
            # Newer artifacts get higher scores
            if age_days <= 30:  # Last month
                score += 1.5
            elif age_days <= 180:  # Last 6 months
                score += 1.0
            elif age_days <= 365:  # Last year
                score += 0.5
            elif age_days > 1825:  # Older than 5 years
                score -= 1.0
        except:
            # If date parsing fails, don't adjust score
            pass
    
    # Ensure score is within bounds
    return max(0.0, min(10.0, score))

def find_location(soup, text):
    """Find the approximate location of text within HTML."""
    # First, check if text appears in page title
    if soup.title and text in soup.title.string:
        return "title"
        
    # Check headings
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        if text in heading.get_text():
            return f"{heading.name}"
            
    # Check code blocks
    for code in soup.find_all(['pre', 'code']):
        if text in code.get_text():
            return "code block"
            
    # Check paragraph text
    for p in soup.find_all('p'):
        if text in p.get_text():
            return "paragraph"
            
    # Check list items
    for li in soup.find_all('li'):
        if text in li.get_text():
            return "list item"
            
    # Check table cells
    for td in soup.find_all('td'):
        if text in td.get_text():
            return "table cell"
            
    # Check div content
    for div in soup.find_all('div'):
        if text in div.get_text():
            return "div"
            
    # Default if location cannot be determined
    return "unknown"

def generate_hash(content):
    """Generate a hash for artifact deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def extract_json_keystores(soup, url, date, artifact_hashes):
    """Extract JSON keystores from HTML."""
    artifacts = []
    
    # Regular expression for JSON keystores (simplified)
    keystore_pattern = re.compile(r'{"crypto":{.*?"ciphertext":.*?}', re.DOTALL)
    
    # Look in <pre>, <code>, and text nodes for JSON keystores
    for element in soup.find_all(['pre', 'code']):
        text = element.get_text()
        
        # Look for JSON keystores
        for match in keystore_pattern.finditer(text):
            keystore_text = match.group(0)
            
            # Generate hash to avoid duplicates
            artifact_hash = generate_hash(keystore_text)
            
            if artifact_hash in artifact_hashes:
                continue
                
            artifact_hashes.add(artifact_hash)
            
            # Score the artifact
            score = score_artifact(url, keystore_text, date)
            
            # Create artifact object
            artifact = {
                'type': 'json_keystore',
                'content': keystore_text,
                'url': url,
                'date': date.isoformat() if date else None,
                'score': score,
                'hash': artifact_hash,
                'location': find_location(soup, keystore_text)
            }
            
            artifacts.append(artifact)
    
    return artifacts

def extract_seed_phrases(soup, url, date, artifact_hashes):
    """Extract seed phrases from HTML."""
    artifacts = []
    
    # Regular expression for seed phrases (12-24 words)
    # This is a simplified version - real detection would be more sophisticated
    seed_pattern = re.compile(r'\b([a-z]{3,8}\s+){11,23}[a-z]{3,8}\b', re.IGNORECASE)
    
    # Extract text from the soup
    text = soup.get_text()
    
    # Look for seed phrases
    for match in seed_pattern.finditer(text):
        seed_phrase = match.group(0)
        
        # Generate hash to avoid duplicates
        artifact_hash = generate_hash(seed_phrase)
        
        if artifact_hash in artifact_hashes:
            continue
            
        artifact_hashes.add(artifact_hash)
        
        # Score the artifact
        score = score_artifact(url, seed_phrase, date)
        
        # Create artifact object
        artifact = {
            'type': 'seed_phrase',
            'content': seed_phrase,
            'url': url,
            'date': date.isoformat() if date else None,
            'score': score,
            'hash': artifact_hash,
            'location': find_location(soup, seed_phrase)
        }
        
        artifacts.append(artifact)
    
    return artifacts

def extract_api_keys(soup, url, date, artifact_hashes):
    """Extract API keys from HTML."""
    artifacts = []
    
    # Patterns for different API key formats
    patterns = {
        'eth_api_key': re.compile(r'eth_apikey[=:]\s*([A-Za-z0-9]{32,})'),
        'etherscan_api_key': re.compile(r'etherscan[_-]?api[_-]?key[=:]\s*([A-Za-z0-9]{32,})'),
        'infura_api_key': re.compile(r'infura[_-]?api[_-]?key[=:]\s*([A-Za-z0-9]{32,})'),
        'general_api_key': re.compile(r'api[_-]?key[=:]\s*([A-Za-z0-9]{16,})')
    }
    
    # Extract text from the soup
    text = soup.get_text()
    
    # Extract API keys of each type
    for key_type, pattern in patterns.items():
        for match in pattern.finditer(text):
            api_key = match.group(1)
            
            # Generate hash to avoid duplicates
            artifact_hash = generate_hash(api_key)
            
            if artifact_hash in artifact_hashes:
                continue
                
            artifact_hashes.add(artifact_hash)
            
            # Score the artifact
            score = score_artifact(url, api_key, date)
            
            # Create artifact object
            artifact = {
                'type': key_type,
                'content': api_key,
                'url': url,
                'date': date.isoformat() if date else None,
                'score': score,
                'hash': artifact_hash,
                'location': find_location(soup, api_key)
            }
            
            artifacts.append(artifact)
    
    return artifacts

def extract_names_from_text(text):
    """
    Extract potential person names from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of extracted names
    """
    # Common name patterns
    # This is a simplified approach - real NER would be more accurate
    name_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b')
    
    # Extract names
    names = name_pattern.findall(text)
    
    # Filter out common non-name capitalized phrases
    filtered_names = []
    for name in names:
        # Skip if it's likely not a name
        if any(word.lower() in ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'it'] for word in name.split()):
            continue
            
        # Skip if it starts with common title words
        if name.split()[0].lower() in ['mr', 'mrs', 'ms', 'miss', 'dr', 'prof']:
            name = ' '.join(name.split()[1:])
            
        filtered_names.append(name)
    
    return filtered_names