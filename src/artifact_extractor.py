#!/usr/bin/env python3
"""
Artifact extraction module for Narrahunt Phase 2.

This module extracts Ethereum-related artifacts from HTML content,
scores them based on various factors, and provides safe outputs.
"""

import re
import os
import json
import hashlib
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Base directory
base_dir = os.path.dirname(os.path.dirname(__file__))

# Ensure output directories exist
os.makedirs(f'{base_dir}/results/artifacts', exist_ok=True)
os.makedirs(f'{base_dir}/results/logs', exist_ok=True)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{base_dir}/results/logs/validation.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('narrahunt.artifact_extractor')

# Whitelist of trusted domains
TRUSTED_DOMAINS = [
    'ethereum.org',
    'ethereum.foundation',
    'eips.ethereum.org',
    'blog.ethereum.org',
    'vitalik.ca'
]

# Community domains (lower score)
COMMUNITY_DOMAINS = [
    'medium.com',
    'hackernoon.com',
    'reddit.com',
    'github.com',
    'steemit.com',
    'mirror.xyz'
]

# Warning phrases that reduce artifact score
WARNING_PHRASES = [
    'do not use in production',
    'example only',
    'not for production',
    'test key',
    'sample key',
    'dummy key',
    'do not use',
    'for testing'
]

# BIP39 word list
logger.info("Loading BIP39 wordlist from file")
wordlist_path = f'{base_dir}/config/wordlists/bip39.txt'

# Ensure the wordlist directory exists
os.makedirs(os.path.dirname(wordlist_path), exist_ok=True)

try:
    with open(wordlist_path, 'r') as f:
        BIP39_WORDS = set(word.strip() for word in f.readlines())
    
    if len(BIP39_WORDS) < 2000:  # BIP39 should have 2048 words
        raise ValueError(f"BIP39 wordlist too small: {len(BIP39_WORDS)} words")
    
    logger.info(f"Loaded {len(BIP39_WORDS)} BIP39 words")
    
except Exception as e:
    logger.error(f"Error loading BIP39 wordlist: {str(e)}")
    logger.warning("Seed phrase detection will be disabled")
    BIP39_WORDS = set()
    
    # Create a minimal wordlist file for future use
    if not os.path.exists(wordlist_path):
        logger.info("Creating minimal BIP39 wordlist for testing")
        minimal_words = ["abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract", "absurd", "abuse"]
        with open(wordlist_path, 'w') as f:
            f.write('\n'.join(minimal_words))
        logger.info(f"Created minimal wordlist at {wordlist_path}")

def extract_artifacts_from_html(html_content, url="", date=None):
    """
    Extract Ethereum artifacts from HTML content.
    
    Args:
        html_content: HTML content to parse
        url: Source URL for scoring
        date: Date of the content for scoring
        
    Returns:
        List of artifact dictionaries
    """
    logger.info(f"Extracting artifacts from URL: {url}")
    
    # Check if html_content is None (non-HTML content)
    if html_content is None:
        logger.warning(f"No HTML content to parse for URL: {url}")
        return []
    
    # Validate content
    content_length = len(html_content) if html_content else 0
    
    # Check for extremely small content
    if content_length < 100:
        logger.warning(f"Content too small ({content_length} bytes) for URL: {url}")
        return []
    
    # Parse HTML
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
    except Exception as e:
        logger.error(f"Error parsing HTML from {url}: {str(e)}")
        return []
    
    # Initialize artifacts list
    artifacts = []
    
    # Track artifact hashes to avoid duplicates
    artifact_hashes = set()
    
    # Process different artifact types
    solidity_artifacts = extract_solidity_contracts(soup, url, date, artifact_hashes)
    artifacts.extend(solidity_artifacts)
    
    wallet_artifacts = extract_wallet_addresses(soup, url, date, artifact_hashes)
    artifacts.extend(wallet_artifacts)
    
    private_key_artifacts = extract_private_keys(soup, url, date, artifact_hashes)
    artifacts.extend(private_key_artifacts)
    
    keystore_artifacts = extract_json_keystores(soup, url, date, artifact_hashes)
    artifacts.extend(keystore_artifacts)
    
    seed_artifacts = extract_seed_phrases(soup, url, date, artifact_hashes)
    artifacts.extend(seed_artifacts)
    
    api_artifacts = extract_api_keys(soup, url, date, artifact_hashes)
    artifacts.extend(api_artifacts)
    
    # Store high-scoring artifacts
    store_artifacts(artifacts)
    
    # Log summary of results
    if artifacts:
        logger.info(f"Extracted {len(artifacts)} artifacts from {url}")
    else:
        logger.info(f"No artifacts found in {url}")
    
    return artifacts

def extract_solidity_contracts(soup, url, date, artifact_hashes):
    """Extract Solidity smart contracts from HTML."""
    artifacts = []
    
    # Find code blocks
    code_blocks = soup.find_all(['pre', 'code'])
    for i, code_block in enumerate(code_blocks):
        code_text = code_block.get_text()
        
        # Look for Solidity contract definitions
        contract_matches = re.finditer(r'contract\s+(\w+)\s*{', code_text)
        for match in contract_matches:
            contract_name = match.group(1)
            
            # Get contract code
            start_pos = match.start()
            # Simple bracket matching to find contract end
            open_braces = 0
            end_pos = start_pos
            
            for j in range(start_pos, len(code_text)):
                if code_text[j] == '{':
                    open_braces += 1
                elif code_text[j] == '}':
                    open_braces -= 1
                    if open_braces == 0:
                        end_pos = j + 1
                        break
            
            if end_pos > start_pos:
                contract_code = code_text[start_pos:end_pos]
                
                # Check for duplicates
                artifact_hash = generate_hash(contract_code)
                if artifact_hash in artifact_hashes:
                    continue
                
                artifact_hashes.add(artifact_hash)
                
                # Score and create artifact
                score = score_artifact(url, contract_code, date)
                
                # Truncate for summary if needed
                summary = contract_code
                if len(summary) > 100:
                    summary = summary[:97] + "..."
                
                artifacts.append({
                    'type': 'solidity_contract',
                    'content': contract_code,
                    'summary': f"Contract {contract_name}: {summary}",
                    'location': f"Code block #{i+1}",
                    'hash': artifact_hash,
                    'score': score,
                    'url': url,
                    'date': date
                })
    
    return artifacts

def extract_wallet_addresses(soup, url, date, artifact_hashes):
    """Extract Ethereum wallet addresses from HTML."""
    artifacts = []
    
    # Search for Ethereum addresses
    text = soup.get_text()
    address_matches = re.finditer(r'0x[0-9a-fA-F]{40}', text)
    
    for match in address_matches:
        address = match.group(0)
        
        # Check for duplicates
        artifact_hash = generate_hash(address)
        if artifact_hash in artifact_hashes:
            continue
        
        artifact_hashes.add(artifact_hash)
        
        # Score and create artifact
        score = score_artifact(url, address, date)
        
        # For addresses, we can show the full content
        artifacts.append({
            'type': 'wallet_address',
            'content': address,
            'summary': address,
            'location': find_location(soup, address),
            'hash': artifact_hash,
            'score': score,
            'url': url,
            'date': date
        })
    
    return artifacts

def extract_private_keys(soup, url, date, artifact_hashes):
    """Extract Ethereum private keys from HTML."""
    artifacts = []
    
    # Search for private keys (64-character hex strings)
    text = soup.get_text()
    private_key_matches = re.finditer(r'(?:private\s*key|secret\s*key|key)(?:\s*[:=])?\s*(?:\'|")?([0-9a-fA-F]{64})(?:\'|")?', text, re.IGNORECASE)
    
    for match in private_key_matches:
        private_key = match.group(1)
        
        # Check for duplicates
        artifact_hash = generate_hash(private_key)
        if artifact_hash in artifact_hashes:
            continue
        
        artifact_hashes.add(artifact_hash)
        
        # Score and create artifact
        score = score_artifact(url, private_key, date)
        
        # For private keys, we must redact the content in the summary
        artifacts.append({
            'type': 'private_key',
            'content': private_key,  # Full content stored for processing
            'summary': f"[private key redacted - {len(private_key)} chars]",
            'location': find_location(soup, private_key),
            'hash': artifact_hash,
            'score': score,
            'url': url,
            'date': date
        })
    
    # Also look for standalone 64-char hex strings
    hex_matches = re.finditer(r'0x[0-9a-fA-F]{64}', text)
    for match in hex_matches:
        hex_string = match.group(0)
        
        # Check for duplicates
        artifact_hash = generate_hash(hex_string)
        if artifact_hash in artifact_hashes:
            continue
        
        artifact_hashes.add(artifact_hash)
        
        # Score and create artifact
        score = score_artifact(url, hex_string, date)
        
        # Redact for summary
        artifacts.append({
            'type': 'private_key',
            'content': hex_string,  # Full content stored for processing
            'summary': f"[private key redacted - {len(hex_string)} chars]",
            'location': find_location(soup, hex_string),
            'hash': artifact_hash,
            'score': score,
            'url': url,
            'date': date
        })
    
    return artifacts

def extract_json_keystores(soup, url, date, artifact_hashes):
    """Extract Ethereum JSON keystores from HTML."""
    artifacts = []
    
    # Find code blocks that might contain JSON
    code_blocks = soup.find_all(['pre', 'code'])
    for i, code_block in enumerate(code_blocks):
        code_text = code_block.get_text()
        
        # Check if it looks like a keystore JSON
        if ('crypto' in code_text.lower() and 
            'cipher' in code_text.lower() and 
            'kdf' in code_text.lower() and 
            'address' in code_text.lower()):
            
            try:
                # Try to parse as JSON
                start_idx = code_text.find('{')
                end_idx = code_text.rfind('}') + 1
                
                if start_idx < 0 or end_idx <= start_idx:
                    continue
                    
                json_text = code_text[start_idx:end_idx]
                
                # Validate JSON format before parsing
                if not ('{' in json_text and '}' in json_text):
                    continue
                    
                try:
                    json_obj = json.loads(json_text)
                except json.JSONDecodeError:
                    # Try to clean up the JSON string and retry
                    json_text = re.sub(r'\\n', '', json_text)
                    json_text = re.sub(r'\\r', '', json_text)
                    json_text = re.sub(r'\\t', '', json_text)
                    json_text = re.sub(r'//.*?\\n', '', json_text)
                    try:
                        json_obj = json.loads(json_text)
                    except:
                        continue
                
                # Verify it's a v3 keystore by checking required fields
                if not ('version' in json_obj and json_obj.get('version') == 3 and
                       'crypto' in json_obj and isinstance(json_obj.get('crypto'), dict)):
                    continue
                
                crypto = json_obj.get('crypto', {})
                if not all(key in crypto for key in ['cipher', 'ciphertext', 'kdf', 'mac']):
                    continue
                
                # Check for duplicates
                artifact_hash = generate_hash(json_text)
                if artifact_hash in artifact_hashes:
                    continue
                
                artifact_hashes.add(artifact_hash)
                
                # Score and create artifact
                score = score_artifact(url, json_text, date)
                
                # Get address safely
                address = json_obj.get('address', '')
                if address and not address.startswith('0x'):
                    address = '0x' + address
                
                # Redact for summary
                artifacts.append({
                    'type': 'ethereum_keystore',
                    'content': json_text,  # Full content stored for processing
                    'summary': f"[JSON keystore redacted] - v3 keystore for address {address}",
                    'location': f"Code block #{i+1}",
                    'hash': artifact_hash,
                    'score': score,
                    'url': url,
                    'date': date
                })
            except Exception as e:
                logger.warning(f"Error processing potential JSON keystore: {str(e)}")
    
    return artifacts

def extract_seed_phrases(soup, url, date, artifact_hashes):
    """Extract BIP39 seed phrases from HTML."""
    artifacts = []
    
    # Find text that might contain seed phrases
    text = soup.get_text()
    
    # Look for sections mentioning mnemonic or seed phrases
    mnemonic_sections = re.finditer(r'(?:mnemonic|seed\s+phrase|recovery\s+phrase|backup\s+phrase)(?:\s*[:=])?\s*(?:\'|")?([a-z\s]+)(?:\'|")?', text, re.IGNORECASE)
    
    for match in mnemonic_sections:
        phrase_text = match.group(1).strip().lower()
        words = phrase_text.split()
        
        # Check if it's a valid length for BIP39 (12, 15, 18, 21, or 24 words)
        if len(words) in [12, 15, 18, 21, 24]:
            # Check if all words are in BIP39 wordlist
            if all(word in BIP39_WORDS for word in words):
                # Check for duplicates
                artifact_hash = generate_hash(phrase_text)
                if artifact_hash in artifact_hashes:
                    continue
                
                artifact_hashes.add(artifact_hash)
                
                # Score and create artifact
                score = score_artifact(url, phrase_text, date)
                
                # Redact for summary
                artifacts.append({
                    'type': 'seed_phrase',
                    'content': phrase_text,  # Full content stored for processing
                    'summary': f"[{len(words)}-word seed phrase redacted]",
                    'location': find_location(soup, phrase_text),
                    'hash': artifact_hash,
                    'score': score,
                    'url': url,
                    'date': date
                })
    
    # Also check for phrases with a specific number of words
    text_blocks = []
    for tag in soup.find_all(['p', 'pre', 'code']):
        text_blocks.append(tag.get_text())
    
    for i, block in enumerate(text_blocks):
        words = block.lower().split()
        if len(words) in [12, 15, 18, 21, 24]:
            # Check if all words are in BIP39 wordlist
            valid_words = [w for w in words if w in BIP39_WORDS]
            if len(valid_words) in [12, 15, 18, 21, 24]:
                phrase_text = ' '.join(valid_words)
                
                # Check for duplicates
                artifact_hash = generate_hash(phrase_text)
                if artifact_hash in artifact_hashes:
                    continue
                
                artifact_hashes.add(artifact_hash)
                
                # Score and create artifact
                score = score_artifact(url, phrase_text, date)
                
                # Redact for summary
                artifacts.append({
                    'type': 'seed_phrase',
                    'content': phrase_text,  # Full content stored for processing
                    'summary': f"[{len(valid_words)}-word seed phrase redacted]",
                    'location': f"Text block #{i+1}",
                    'hash': artifact_hash,
                    'score': score,
                    'url': url,
                    'date': date
                })
    
    return artifacts

def extract_api_keys(soup, url, date, artifact_hashes):
    """Extract API keys (Infura, Alchemy, Etherscan) from HTML."""
    artifacts = []
    
    # Find code blocks
    code_blocks = soup.find_all(['pre', 'code'])
    for i, code_block in enumerate(code_blocks):
        code_text = code_block.get_text()
        
        # Look for Infura endpoints
        infura_matches = re.finditer(r'https?://[^"\']*infura\.io/v3/([0-9a-fA-F]{32})', code_text)
        for match in infura_matches:
            api_key = match.group(1)
            
            # Check for duplicates
            artifact_hash = generate_hash(api_key)
            if artifact_hash in artifact_hashes:
                continue
            
            artifact_hashes.add(artifact_hash)
            
            # Score and create artifact
            score = score_artifact(url, api_key, date)
            
            # Redact for summary
            artifacts.append({
                'type': 'api_key',
                'content': api_key,  # Full content stored for processing
                'summary': f"[Infura API key redacted - {len(api_key)} chars]",
                'location': f"Code block #{i+1}",
                'hash': artifact_hash,
                'score': score,
                'url': url,
                'date': date
            })
        
        # Look for Alchemy endpoints
        alchemy_matches = re.finditer(r'https?://[^"\']*alchemy\.com/v2/([0-9a-zA-Z]{32,})', code_text)
        for match in alchemy_matches:
            api_key = match.group(1)
            
            # Check for duplicates
            artifact_hash = generate_hash(api_key)
            if artifact_hash in artifact_hashes:
                continue
            
            artifact_hashes.add(artifact_hash)
            
            # Score and create artifact
            score = score_artifact(url, api_key, date)
            
            # Redact for summary
            artifacts.append({
                'type': 'api_key',
                'content': api_key,  # Full content stored for processing
                'summary': f"[Alchemy API key redacted - {len(api_key)} chars]",
                'location': f"Code block #{i+1}",
                'hash': artifact_hash,
                'score': score,
                'url': url,
                'date': date
            })
        
        # Look for Etherscan API keys
        etherscan_matches = re.finditer(r'(?:etherscan|ETHERSCAN).*?(?:apikey|ApiKey|APIKEY).*?[\'"]([A-Za-z0-9]{34,})[\'"]', code_text)
        for match in etherscan_matches:
            api_key = match.group(1)
            
            # Check for duplicates
            artifact_hash = generate_hash(api_key)
            if artifact_hash in artifact_hashes:
                continue
            
            artifact_hashes.add(artifact_hash)
            
            # Score and create artifact
            score = score_artifact(url, api_key, date)
            
            # Redact for summary
            artifacts.append({
                'type': 'api_key',
                'content': api_key,  # Full content stored for processing
                'summary': f"[Etherscan API key redacted - {len(api_key)} chars]",
                'location': f"Code block #{i+1}",
                'hash': artifact_hash,
                'score': score,
                'url': url,
                'date': date
            })
    
    return artifacts

def score_artifact(url, content, date):
    """
    Score an artifact based on source and content.
    
    Args:
        url: Source URL
        content: Artifact content
        date: Content date
        
    Returns:
        Score value (int)
    """
    score = 0
    
    # Domain scoring
    domain = urlparse(url).netloc
    
    # +3 for trusted domains
    if any(domain.endswith(trusted) for trusted in TRUSTED_DOMAINS):
        score += 3
    
    # +1 for .org TLD
    if domain.endswith('.org'):
        score += 1
    
    # -5 for community domains
    if any(domain.endswith(community) for community in COMMUNITY_DOMAINS):
        score -= 5
    
    # Date scoring
    if date and date < "2022-01-01":
        # +1 for pre-2022 snapshot
        score += 1
    
    # Content scoring
    content_lower = content.lower()
    
    # -10 for warning phrases
    if any(warning in content_lower for warning in WARNING_PHRASES):
        score -= 10
    
    # +2 for syntactically valid artifacts
    # Check for specific artifact validity based on type
    if 'contract ' in content_lower and '{' in content and '}' in content:
        # Likely a valid Solidity contract
        score += 2
    elif re.match(r'^0x[0-9a-f]{40}$', content_lower):
        # Valid Ethereum address format
        score += 2
    elif re.match(r'^0x[0-9a-f]{64}$', content_lower) or re.match(r'^[0-9a-f]{64}$', content_lower):
        # Valid private key format
        score += 3
    elif re.match(r'^(?:\w+\s){11,23}\w+$', content_lower) and len(content.split()) in [12, 15, 18, 21, 24]:
        # Potentially valid seed phrase format
        score += 2
    elif re.match(r'^[A-Za-z0-9]{32,}$', content):
        # Potentially valid API key
        score += 1
    
    # Ensure score is within valid bounds
    score = max(0.0, min(10.0, score))  # Cap at 10 instead of unbounded
    return score

def find_location(soup, text):
    """Find location of text in the HTML with contextual information."""
    # Check for the text in various elements with specific handling
    
    # First try to find exact matches with their parent context
    for element_type in ['pre', 'code', 'p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a']:
        elements = soup.find_all(element_type)
        for i, element in enumerate(elements):
            if text in element.get_text():
                # Try to get parent context for better location info
                parent = element.parent
                parent_id = parent.get('id', '')
                parent_class = ' '.join(parent.get('class', []))
                
                location = f"{element_type}#{i+1}"
                
                # Add additional context if available
                if parent_id:
                    location += f" in div#{parent_id}"
                elif parent_class:
                    location += f" in div.{parent_class}"
                
                # Try to get section heading
                heading = None
                for heading_tag in ['h1', 'h2', 'h3', 'h4']:
                    # Look for previous heading
                    prev_heading = element.find_previous(heading_tag)
                    if prev_heading:
                        heading = prev_heading.get_text().strip()
                        break
                
                if heading:
                    location += f" under '{heading[:30]}...'" if len(heading) > 30 else f" under '{heading}'"
                
                return location
    
    # For more structured documents, try to determine section by tree traversal
    all_elements = soup.find_all()
    for i, element in enumerate(all_elements):
        if text in element.get_text():
            # Get a path-like description
            parents = []
            parent = element.parent
            # Limit to 3 levels of parents to avoid overly long paths
            for _ in range(3):
                if parent and parent.name != '[document]':
                    parent_id = parent.get('id', '')
                    if parent_id:
                        parents.append(f"{parent.name}#{parent_id}")
                    else:
                        parents.append(parent.name)
                    parent = parent.parent
                else:
                    break
            
            path = ' > '.join(reversed(parents))
            if path:
                return f"{element.name} in {path}"
            else:
                return f"{element.name}#{i+1}"
    
    return "Unknown location"

def generate_hash(content):
    """Generate a unique hash for artifact deduplication."""
    # Normalize by removing whitespace and converting to lowercase
    normalized = re.sub(r'\s+', '', content.lower())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def store_artifacts(artifacts):
    """Store high-scoring artifacts."""
    today = datetime.now().strftime('%Y-%m-%d')
    artifacts_dir = f'{base_dir}/results/artifacts/{today}'
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Store high-scoring artifacts in found.txt
    with open(f'{base_dir}/results/found.txt', 'a') as found_file:
        for artifact in artifacts:
            if artifact['score'] > 0:
                # Store as JSON
                artifact_path = f"{artifacts_dir}/{artifact['hash']}.json"
                
                # Create a safe version for storage
                safe_artifact = artifact.copy()
                
                # For sensitive artifacts, replace content with a hash reference
                if artifact['type'] in ['private_key', 'seed_phrase', 'api_key']:
                    # Replace the actual content with a reference
                    safe_artifact['content_hash'] = artifact['hash']
                    safe_artifact.pop('content', None)
                
                with open(artifact_path, 'w') as f:
                    json.dump(safe_artifact, f, indent=2)
                
                # Log to found.txt
                found_file.write(f"URL: {artifact['url']}\n")
                found_file.write(f"Type: {artifact['type']}\n")
                found_file.write(f"Score: {artifact['score']}\n")
                found_file.write(f"Location: {artifact['location']}\n")
                found_file.write(f"Summary: {artifact['summary']}\n")
                found_file.write(f"File: {artifact_path}\n")
                found_file.write("-" * 80 + "\n")
    
    return len([a for a in artifacts if a['score'] > 0])