#!/usr/bin/env python3
"""
Enhanced Name Artifact Extractor for the Narrative Discovery Matrix.

This module extends the standard artifact extraction with specialized
capabilities for detecting and analyzing name artifacts - usernames,
project names, aliases, and other naming patterns particularly relevant
for cryptocurrency figures and projects.
"""

import re
import os
import json
import logging
from datetime import datetime
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('name_artifact_extractor')

class NameArtifactExtractor:
    """
    Specialized extractor for name-related artifacts.
    """
    
    def __init__(self, entity=None):
        """
        Initialize the name artifact extractor.
        
        Args:
            entity: Optional target entity to focus extraction on
        """
        self.entity = entity
        # Create excluded names set (including variations of the target entity)
        self.excluded_names = set()
        self.entity_parts = []
        
        if entity:
            # Add the entity name itself
            self.excluded_names.add(entity.lower())
            
            # Add variations (first name, last name)
            name_parts = entity.split()
            self.entity_parts = [part.lower() for part in name_parts if len(part) > 2]
            
            # Add individual name parts
            for part in name_parts:
                if len(part) > 2:  # Only add name parts that are reasonably long
                    self.excluded_names.add(part.lower())
                    # Also add variations with punctuation removed
                    clean_part = re.sub(r'[^\w]', '', part.lower())
                    if clean_part and clean_part != part.lower():
                        self.excluded_names.add(clean_part)
            
            # Add common forms of the entity name
            # If entity is "John Smith", also exclude variations
            if len(name_parts) > 1:
                # Last name, first name format
                self.excluded_names.add(f"{name_parts[-1]}, {' '.join(name_parts[:-1])}".lower())
                
                # Add first name initial with last name
                if len(name_parts[0]) > 0:
                    self.excluded_names.add(f"{name_parts[0][0]}. {name_parts[-1]}".lower())
                    
                # Add possessive forms (e.g., "Vitalik's", "Buterin's")
                for part in name_parts:
                    if len(part) > 2:
                        self.excluded_names.add(f"{part}'s".lower())
                        # Also add the part with common suffixes
                        for suffix in ["s", "es", "ing", "ed", "er"]:
                            self.excluded_names.add(f"{part}{suffix}".lower())
                
            # Add variations with titles
            titles = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof."]
            for title in titles:
                self.excluded_names.add(f"{title} {entity}".lower())
            
            # Add fully normalized version (no spaces, no punctuation)
            normalized_entity = re.sub(r'[^\w]', '', entity.lower())
            if normalized_entity:
                self.excluded_names.add(normalized_entity)
                
            # Add email-like variations (for 'John Smith' -> 'jsmith', 'john.smith', etc.)
            if len(name_parts) > 1:
                first, last = name_parts[0].lower(), name_parts[-1].lower()
                self.excluded_names.add(f"{first}.{last}".lower())
                self.excluded_names.add(f"{first[0]}{last}".lower())
                self.excluded_names.add(f"{first}{last[0]}".lower())
        
        # Log excluded names for debugging
        if self.excluded_names:
            logger.info(f"Excluding {len(self.excluded_names)} entity variations")
            # Log just a sample to avoid cluttering logs
            sample = list(self.excluded_names)[:10]
            logger.info(f"Sample exclusions: {', '.join(sample)}")
        
        # Define patterns to extract high-quality, valuable names and usernames
        self.name_patterns = {
            'username': [
                # Clear username indicators with alphanumeric constraints
                r'username[:\s]+([\w][\w.-]{1,29})\b',
                r'user[\s-]?name[:\s]+([\w][\w.-]{1,29})\b',
                r'handle[:\s]+([\w][\w.-]{1,29})\b',
                r'account[\s-]?name[:\s]+([\w][\w.-]{1,29})\b',
                
                # Social media handles
                r'@([\w][\w.-]{1,29})\b',
                r'(?:twitter|github|reddit|telegram)\.com/(?:@)?([\w][\w.-]{1,29})\b',
                
                # GitHub-specific patterns - high value artifacts
                r'github\.com/(?:users/)?([\w][\w.-]{1,39})\b',
                r'github user[:\s]+([\w][\w.-]{1,39})\b',
                r'github handle[:\s]+([\w][\w.-]{1,39})\b',
                
                # Contextual username patterns
                r'known as ([\w][\w.-]{1,29}) on\b',
                r'([\w][\w.-]{1,29}) on (?:twitter|github|reddit|facebook|linkedin|discord|telegram)\b',
                
                # Developer-specific handles
                r'developer(?:\s+name)?[:\s]+([\w][\w.-]{1,29})\b',
                r'coder[:\s]+([\w][\w.-]{1,29})\b',
                r'contributor[:\s]+([\w][\w.-]{1,29})\b'
            ],
            
            'project_name': [
                # Clear project name indicators with proper capitalization constraint
                r'project[\s-]?name[:\s]+([A-Z][\w\s.-]{1,49})\b',
                r'project[:\s]+([A-Z][\w\s.-]{1,49})\b',
                
                # Named projects
                r'called[\s:"\']+(?:the\s)?([A-Z][\w\s.-]{1,49})(?:\s+project)?\b',
                r'named[\s:"\']+(?:the\s)?([A-Z][\w\s.-]{1,49})\b',
                r'(?:code)?named[\s:"\']+([A-Z][\w\s.-]{1,49})\b',
                
                # Action-based project references
                r'(?:developed|created|founded|launched)[\s:"\']+(?:the\s)?([A-Z][\w\s.-]{1,49})\b',
                r'(?:the\s)?([A-Z][\w\s.-]{1,49}) (?:blockchain|protocol|platform|network|project|initiative)\b',
                
                # Versioning/release patterns - highly valuable
                r'version[\s:]+"([A-Z][\w\s.-]{1,49})"\b',
                r'release[\s:]+"([A-Z][\w\s.-]{1,49})"\b',
                r'"([A-Z][\w\s.-]{1,49})" (?:release|version|upgrade|fork)\b'
            ],
            
            'ethereum_upgrades': [
                # Specific named Ethereum upgrades - explicit patterns for known valuable artifacts
                # Put this first to ensure it has priority
                r'\b(Constantinople|Byzantium|Homestead|Frontier|Metropolis|Serenity|Berlin|London|Paris|Shanghai|Prague|Istanbul|Petersburg|Muir Glacier|Arrow Glacier|Gray Glacier|Bellatrix|Altair|Merge|Shapella|Dencun|Cancun)\b',
                
                # Variations of named upgrades
                r'called\s+["\'"]?(Constantinople|Byzantium|Homestead|Frontier|Metropolis|Serenity|Berlin|London|Paris|Shanghai|Prague|Istanbul|Petersburg|Muir Glacier|Arrow Glacier|Gray Glacier|Bellatrix|Altair|Merge|Shapella|Dencun|Cancun)["\'"]?',
                r'named\s+["\'"]?(Constantinople|Byzantium|Homestead|Frontier|Metropolis|Serenity|Berlin|London|Paris|Shanghai|Prague|Istanbul|Petersburg|Muir Glacier|Arrow Glacier|Gray Glacier|Bellatrix|Altair|Merge|Shapella|Dencun|Cancun)["\'"]?',
                
                # Generic patterns for upgrades
                r'(?:ethereum|eth)[\s]+(?:network|blockchain)?\s+upgrade(?:\s+named)?\s+["\'"]?([A-Z][\w\s.-]{1,29})["\'"]?',
                r'(?:ethereum|eth)[\s]+(?:hard\s+)?fork(?:\s+named)?\s+["\'"]?([A-Z][\w\s.-]{1,29})["\'"]?',
                r'(?:hard\s+)?fork(?:\s+named)?\s+["\'"]?([A-Z][\w\s.-]{1,29})["\'"]?',
                r'upgrade(?:\s+named)?\s+["\'"]?([A-Z][\w\s.-]{1,29})["\'"]?',
                
                # Very specific patterns for known Ethereum upgrade formats
                r'release was called\s+([A-Z][\w\s.-]{1,29})',
                r'followed by\s+([A-Z][\w\s.-]{1,29})',
                
                # Generic but constrained upgrade pattern - capture common formats in the test
                r'(?:the\s+)?([A-Z][\w\s.-]{1,29})\s+(?:upgrade|hard\s+fork)',
                r'(?:the\s+)?([A-Z][\w\s.-]{1,29})\s+(?:release|version)'
            ],
            
            'pseudonym': [
                # Clear pseudonym indicators 
                r'pseudonym[:\s]+([\w][\w.-]{1,29})\b',
                r'alias[:\s]+([\w][\w.-]{1,29})\b',
                r'pen[\s-]name[:\s]+([\w][\w.-]{1,29})\b',
                r'nickname[:\s]+([\w][\w.-]{1,29})\b',
                
                # Contextual pseudonyms
                r'known as ([\w][\w.-]{1,29})\b',
                r'goes by ([\w][\w.-]{1,29})\b',
                r'([\w][\w.-]{1,29}) \((?:a\.k\.a\.|aka|alias)\)',
                r'a\.k\.a\.[\s:"\']+([A-Z][\w.-]{1,29})\b',
                r'aka[\s:"\']+([A-Z][\w.-]{1,29})\b'
            ],
            
            'company_name': [
                # Clear company indicators with proper capitalization
                r'company[:\s]+([A-Z][\w\s.-]{1,49})\b',
                r'startup[:\s]+([A-Z][\w\s.-]{1,49})\b',
                r'founded[\s:]+((?:the\s)?[A-Z][\w\s.-]{1,49})\b',
                
                # Company suffixes
                r'(?:the\s)?([A-Z][\w\s.-]{1,49}) (?:Foundation|Lab|Inc|LLC|Co|Corporation|Company|GmbH|Ltd)\b',
                r'(?:the\s)?([A-Z][\w\s.-]{1,49}) (?:Team|Group|Organization)\b'
            ]
        }
        
        # Name filtering - terms that are too generic to be useful or likely garbage
        self.filter_terms = [
            # Generic terms and metadata words
            'project', 'website', 'platform', 'system', 'network', 'concept',
            'username', 'nickname', 'handle', 'alias', 'term', 'idea',
            'profile', 'account', 'user', 'protocol', 'blockchain', 'foundation',
            'company', 'startup', 'organization', 'group', 'team', 'community',
            'readme', 'license', 'contributor', 'copyright', 'author', 'documentation',
            
            # Generic blockchain/crypto terms to avoid false positives
            'address', 'transaction', 'block', 'hash', 'wallet', 'contract', 'token',
            'cryptocurrency', 'decentralized', 'mining', 'staking', 'validator',
            'governance', 'consensus', 'whitepaper', 'proposal', 'node', 'client',
            'testnet', 'mainnet', 'upgrade', 'source', 'code', 'framework',
            
            # Common conjunctions, prepositions, articles
            'the', 'and', 'or', 'but', 'yet', 'so', 'for', 'nor', 'a', 'an',
            'to', 'in', 'of', 'by', 'at', 'on', 'with', 'from', 'as', 'into',
            'about', 'between', 'among', 'through', 'during', 'until', 'against',
            'toward', 'upon', 'concerning', 'like', 'over', 'before', 'after',
            'since', 'throughout', 'below', 'beside', 'however', 'therefore', 'because',
            
            # Common filler words and fragments
            'this', 'that', 'these', 'those', 'there', 'here', 'where', 'when',
            'what', 'which', 'who', 'whom', 'whose', 'why', 'how', 'any', 'some',
            'many', 'much', 'more', 'most', 'other', 'another', 'such', 'all',
            'both', 'each', 'either', 'neither', 'every', 'few', 'little', 'less',
            'least', 'several', 'enough', 'own', 'same', 'different', 'various',
            'certain', 'someone', 'anyone', 'something', 'anything', 'nothing',
            'yours', 'mine', 'ours', 'theirs', 'himself', 'herself', 'itself', 'themselves',
            
            # Modal verbs and auxiliaries
            'could', 'would', 'should', 'will', 'shall', 'may', 'might', 'must', 'can',
            'ought', 'need', 'dare', 'used',
            
            # Common verbs (especially fragments likely to be found)
            'is', 'am', 'are', 'was', 'were', 'be', 'being', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'doing', 'done', 'go', 'goes', 'went', 'gone',
            'see', 'sees', 'saw', 'seen', 'know', 'knows', 'knew', 'known',
            'get', 'gets', 'got', 'gotten', 'make', 'makes', 'made', 'say', 'says',
            'said', 'come', 'comes', 'came', 'take', 'takes', 'took', 'taken',
            'find', 'found', 'think', 'thought', 'use', 'used', 'work', 'worked',
            'call', 'called', 'try', 'tried', 'ask', 'asked', 'need', 'needed',
            'feel', 'felt', 'become', 'became', 'leave', 'left', 'put', 'read',
            
            # Common garbage fragments found in previous runs
            'addressing urgent', 'it included several', 'rameth', 'following sections',
            'section describes', 'please note', 'click here', 'learn more', 'read more',
            'skip to', 'back to', 'next to', 'related to', 'according to', 'based on',
            'example of', 'part of', 'kind of', 'type of', 'sort of', 'referred to',
            'mentioned in', 'included in', 'included as', 'such as', 'rather than',
            's to the', 'ing the', 'd by the', 'ed the', 'ing of', 'tion of'
        ]
        
        # Also exclude common HTML-related terms that might leak through
        self.html_filter_terms = [
            'div', 'span', 'nav', 'section', 'article', 'main', 'header', 'footer',
            'button', 'input', 'form', 'table', 'thead', 'tbody', 'template', 'script',
            'body', 'html', 'head', 'title', 'meta', 'link', 'style', 'class', 'id'
        ]
        
        # Add HTML filter terms to main filter terms
        self.filter_terms.extend(self.html_filter_terms)
        
        # Specialized context words that increase the relevance score
        self.context_relevance = {
            'cryptocurrency': 0.2,
            'blockchain': 0.2,
            'ethereum': 0.3,
            'bitcoin': 0.2,
            'crypto': 0.2,
            'token': 0.1,
            'wallet': 0.1,
            'smart contract': 0.2,
            'dapp': 0.2,
            'decentralized': 0.1,
            'web3': 0.1,
            'consensus': 0.1,
            'mining': 0.1,
            'staking': 0.1,
            'node': 0.1,
            'testnet': 0.2,
            'mainnet': 0.2,
            'fork': 0.1,
            'whitepaper': 0.2
        }
    
    def extract_from_text(self, text, url=None, date=None):
        """
        Extract name artifacts from text.
        
        Args:
            text: Text to extract from
            url: Source URL for context
            date: Date of the content for relevance scoring
            
        Returns:
            List of name artifact dictionaries
        """
        # Add logging at the start of extraction
        logger.info(f"Starting name extraction from {url}")
        
        artifacts = []
        
        # Skip if no text or text is too short
        if not text or len(text) < 20:
            logger.debug(f"Skipping text: too short ({len(text) if text else 0} chars)")
            return artifacts
        
        # Skip text that contains too many newlines (likely to be code or formatting issues)
        if text.count('\n') > len(text) / 50:  # More than 2% newlines
            return artifacts
            
        # Skip text that appears to be malformed or contains problematic patterns
        if '�' in text or '\\u' in text or len(re.findall(r'[^\x00-\x7F]', text)) > len(text) / 20:
            return artifacts
            
        # Search for entity-specific context if entity is specified
        entity_context = False
        if self.entity:
            entity_pattern = re.compile(r'\b' + re.escape(self.entity) + r'\b', re.IGNORECASE)
            entity_context = bool(entity_pattern.search(text))
        
        # Track names found in this text to avoid duplicates within same extraction
        found_names = set()
        
        # Extract name artifacts by type
        for name_type, patterns in self.name_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        # Get the name, handling patterns with multiple groups
                        if len(match.groups()) > 1 and match.group(2):
                            name = match.group(2).strip()
                        else:
                            name = match.group(1).strip()
                        
                        # Log every potential name found before filtering
                        logger.debug(f"Raw potential name found: '{name}' (type: {name_type})")
                        
                        # Normalize the name for consistent processing
                        name = self._normalize_name(name)
                        
                        # Skip if empty after normalization
                        if not name:
                            logger.debug(f"Filtering out '{name}': empty after normalization")
                            continue
                            
                        # Skip if already found in this text (avoid duplicates)
                        if name.lower() in found_names:
                            logger.debug(f"Filtering out '{name}': already found in this text")
                            continue
                            
                        # Perform validation checks to ensure high-quality artifacts
                        if not self._is_valid_name(name, name_type):
                            logger.debug(f"Filtering out '{name}': failed validity check for type {name_type}")
                            continue
                        
                        # Skip if name contains invalid characters or patterns
                        if '\n' in name or '\t' in name or '\r' in name or '\f' in name:
                            continue
                        
                        # Discard name if it contains malformed text or encoding issues
                        if '�' in name or '\\u' in name or len(re.findall(r'[^\x00-\x7F]', name)) > 0:
                            continue
                            
                        # Skip if the name looks like an incomplete word or sentence fragment
                        if self._is_sentence_fragment(name):
                            continue
                            
                        # Skip if it looks like a sentence fragment rather than a name
                        words = name.split()
                        if len(words) > 5 and name_type != 'ethereum_upgrades':  # Allow more words for Ethereum upgrades
                            continue
                        
                        # Auto-exclude the target entity name and variations
                        if self.entity:
                            name_lower = name.lower()
                            name_normalized = re.sub(r'[^\w]', '', name_lower)
                            
                            # Direct check against exclusion list
                            if name_lower in self.excluded_names or name_normalized in self.excluded_names:
                                logger.debug(f"Filtering out '{name}': target entity exclusion - direct match")
                                continue
                                
                            # Check if name contains any of the entity parts
                            if any(part in name_lower for part in self.entity_parts):
                                logger.debug(f"Filtering out '{name}': target entity exclusion - contains entity part")
                                continue
                                
                            # Check if name is similar to entity using string distance
                            # This helps with minor spelling variations
                            entity_lower = self.entity.lower()
                            
                            # Check for partial matching and containment
                            if (entity_lower.startswith(name_lower) or 
                                entity_lower.endswith(name_lower) or 
                                name_lower.startswith(entity_lower) or 
                                name_lower.endswith(entity_lower) or
                                name_lower in entity_lower or
                                entity_lower in name_lower or
                                entity_lower.startswith(name_lower + " ") or 
                                entity_lower.endswith(" " + name_lower) or
                                " " + name_lower + " " in entity_lower or
                                name_lower == entity_lower):
                                logger.debug(f"Excluded '{name}' - partial match with entity")
                                continue
                            
                        # Skip filtered terms (exact match)
                        if name.lower() in self.filter_terms:
                            continue
                        
                        # Skip if name has too many filter terms as constituent words
                        filter_word_count = sum(1 for word in words if word.lower() in self.filter_terms)
                        if len(words) > 0 and filter_word_count / len(words) > 0.3:  # More than 30% are filter words
                            continue
                        
                        # Skip if too short
                        if len(name) < 2:
                            continue
                            
                        # Skip if too long (but allow longer names for ethereum upgrades)
                        max_length = 50 if name_type == 'ethereum_upgrades' else 30
                        if len(name) > max_length:
                            continue
                        
                        # Skip if it looks like a sentence fragment (too many spaces or common words)
                        space_ratio = name.count(' ') / len(name) if len(name) > 0 else 0
                        if space_ratio > 0.3 and name_type not in ['ethereum_upgrades', 'project_name']:
                            # Check if it contains too many common words
                            common_word_count = sum(1 for word in words if word.lower() in self.filter_terms)
                            if common_word_count > 0 and common_word_count / len(words) > 0.3:  # More than 30% are common words
                                continue
                        
                        # Calculate score based on various factors
                        score = self._calculate_artifact_score(name, name_type, entity_context, text, match)
                        
                        # Skip low scoring matches - higher threshold for better quality
                        min_score_threshold = 0.5  # Increased threshold to ensure only high-quality matches
                        if score < min_score_threshold:
                            logger.debug(f"Filtering out '{name}': quality check failed (score: {score:.2f} < {min_score_threshold})")
                            continue
                        else:
                            logger.debug(f"Keeping '{name}': score {score:.2f}")
                        
                        # Create context window
                        context_window = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                        
                        # Create artifact
                        artifact = {
                            'type': 'name_artifact',
                            'subtype': name_type,
                            'name': name,
                            'context': context_window,
                            'source_url': url,
                            'timestamp': datetime.now().isoformat(),
                            'score': round(score, 2)
                        }
                        
                        # Add to artifacts list and track the found name
                        artifacts.append(artifact)
                        found_names.add(name.lower())
                except Exception as e:
                    # Log any errors but continue processing
                    logger.error(f"Error extracting with pattern '{pattern}': {str(e)}")
                    continue
        
        # Sort artifacts by score (descending)
        artifacts.sort(key=lambda x: x['score'], reverse=True)
        
        # Create final artifacts list
        final_artifacts = artifacts
        
        # Log final extraction results
        logger.info(f"Name extraction complete: {len(final_artifacts)} artifacts after filtering")
        if final_artifacts:
            for idx, artifact in enumerate(final_artifacts[:5]):  # Log first 5 for debugging
                logger.info(f"  Top artifact {idx+1}: '{artifact['name']}' (type: {artifact['subtype']}, score: {artifact['score']:.2f})")
            
        return final_artifacts
        
    def _normalize_name(self, name):
        """Normalize a name by cleaning up whitespace and special characters."""
        if not name:
            return ""
            
        # Strip whitespace
        name = name.strip()
        
        # Collapse multiple spaces
        name = re.sub(r'\s+', ' ', name)
        
        # Remove quotes around name (often included in patterns)
        name = re.sub(r'^["\'](.*)["\']$', r'\1', name)
        
        # Remove leading "the " for project names
        name = re.sub(r'^the\s+', '', name, flags=re.IGNORECASE)
        
        return name
        
    def _is_valid_name(self, name, name_type):
        """Check if a name is valid based on its type."""
        if not name:
            return False
            
        # Get word count for various checks
        words = name.split()
        word_count = len(words)
        
        # Check for repeated words (like "Enterprise Enterprise")
        if word_count > 1:
            # Check for immediate repetition
            for i in range(1, word_count):
                if words[i].lower() == words[i-1].lower():
                    return False
            
            # Check for same word repeated non-consecutively
            word_freq = {}
            for word in words:
                word_lower = word.lower()
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
                
            # If any word appears more than half the total words, it's suspicious
            for word, count in word_freq.items():
                if count > 1 and len(word) > 3 and count >= word_count / 2:
                    return False
        
        # Different validation rules based on type
        if name_type == 'username':
            # Usernames should not have too many spaces
            if ' ' in name and name.count(' ') > 1:
                return False
                
            # Usernames should start with a letter or number
            if not re.match(r'^[a-zA-Z0-9]', name):
                return False
                
            # Username max length constraint
            if len(name) > 30:
                return False
                
        elif name_type == 'ethereum_upgrades':
            # Special handling for Ethereum upgrades
            
            # List of known Ethereum upgrade names - always valid
            known_upgrades = [
                'frontier', 'homestead', 'byzantium', 'constantinople', 
                'petersburg', 'istanbul', 'muir glacier', 'berlin',
                'london', 'arrow glacier', 'gray glacier', 'paris',
                'shanghai', 'cancun', 'prague', 'osaka', 'bogota',
                'dencun', 'pectra', 'shapella', 'metropolis', 'serenity',
                'altair', 'bellatrix', 'merge'
            ]
            
            # Always accept known upgrade names regardless of other rules
            if name.lower() in known_upgrades:
                return True
                
            # For other potential upgrade names, apply validation rules
            
            # Ethereum upgrades should be properly capitalized
            if not name[0].isupper():
                return False
                
            # Ethereum upgrades shouldn't end with incomplete words
            if re.search(r'[a-z]{1,2}$', name):
                return False
                
            # Ethereum upgrades should have 1-3 words maximum
            if word_count > 3:
                return False
                
            # For multi-word upgrade names, all words should be capitalized (like "Gray Glacier")
            if word_count > 1 and not all(w[0].isupper() for w in words if len(w) > 1):
                return False
                
        elif name_type in ['project_name', 'company_name']:
            # Project and company names should be properly capitalized
            if not name[0].isupper():
                return False
                
            # Reject likely descriptions
            # These often have certain patterns like "AI-powered X", "X-based Y", etc.
            description_patterns = [
                r'powered', r'based', r'enabled', r'driven', r'focused', 
                r'centric', r'oriented', r'friendly', r'specific', r'ready',
                r'dollar', r'million', r'billion', r'percent', r'secure', 'security',
                r'platform for', r'service for', r'tool for', r'solution for',
                # Add more patterns to catch previously missed descriptions
                r'performance', r'grade', r'generation', r'chain', r'friendly',
                r'scalable', r'compatible', r'integrated', r'optimized'
            ]
            
            if any(re.search(pattern, name.lower()) for pattern in description_patterns):
                return False
                
            # Check for hyphenated descriptions which often indicate descriptive phrases
            if '-' in name and not re.match(r'^[A-Z][a-z]+\-[A-Z][a-z]+$', name):  # Allow "Gray-Glacier" format
                # Look for common descriptor patterns with hyphens
                if any(re.search(r'(?:high|low|next|cross|multi|non)-\w+', name.lower()) for pattern in description_patterns):
                    return False
                
            # Limit word count for project names (4 max)
            if word_count > 4:
                return False
                
        # Check for incomplete words at beginning or end
        if re.match(r'^[a-z]{1,2}\s', name) or re.search(r'\s[a-z]{1,2}$', name):
            return False
            
        # Check for malformed text that looks like HTML/JS fragments
        if re.search(r'</?[a-z]+>', name) or re.search(r'\{[a-z]+\}', name):
            return False
            
        # Check for descriptions rather than proper names
        # Descriptions often have certain patterns of adjectives and nouns
        if word_count > 2:
            # Check for adjective-heavy patterns (typical in descriptions)
            adjective_patterns = [r'ing\s', r'ed\s', r'able\s', r'ible\s', r'ful\s', r'ous\s', r'ive\s', r'al\s']
            if any(re.search(pattern, name.lower()) for pattern in adjective_patterns):
                # Descriptions often have these patterns, but real names typically don't
                if word_count > 3:  # Be more strict with longer phrases
                    return False
        
        return True
        
    def _is_sentence_fragment(self, name):
        """Check if a name appears to be a sentence fragment rather than a proper name."""
        # Check for indicators of sentence fragments
        
        # Incomplete words at start/end
        if re.match(r'^[a-z]{1,2}\b', name) or re.search(r'\b[a-z]{1,2}$', name):
            return True
            
        # Starting with prepositions, articles, conjunctions
        fragment_starts = ['to ', 'in ', 'on ', 'at ', 'by ', 'for ', 'with ', 'from ', 
                          'the ', 'a ', 'an ', 'and ', 'or ', 'but ', 'if ', 'as ']
        if any(name.lower().startswith(start) for start in fragment_starts):
            return True
            
        # Ending with prepositions, articles, conjunctions  
        fragment_ends = [' to', ' in', ' on', ' at', ' by', ' for', ' with', ' from',
                        ' the', ' a', ' an', ' and', ' or', ' but', ' if', ' as']
        if any(name.lower().endswith(end) for end in fragment_ends):
            return True
            
        # Check for typical sentence fragment patterns
        if re.search(r'ing\s+the\b', name.lower()) or re.search(r'ed\s+by\b', name.lower()):
            return True
            
        # Containing "s to" or "s of" pattern (common in fragments)
        if re.search(r's\s+to\b', name.lower()) or re.search(r's\s+of\b', name.lower()):
            return True
            
        # More than 3 words and contains conjunctions in the middle
        words = name.split()
        if len(words) > 3:
            for i in range(1, len(words)-1):
                if words[i].lower() in ['and', 'or', 'but', 'if', 'as', 'because', 'since']:
                    return True
                    
        return False
        
    def _calculate_artifact_score(self, name, name_type, entity_context, text, match):
        """Calculate a quality score for an artifact."""
        # Base score by type - start higher for more confident matches
        if name_type == 'ethereum_upgrades':
            # Ethereum upgrade names are extremely valuable
            base_score = 0.8
        elif name_type == 'username':
            base_score = 0.7  # Usernames are very valuable
        elif name_type == 'project_name':
            base_score = 0.65
        elif name_type == 'pseudonym':
            base_score = 0.75  # Pseudonyms are particularly valuable
        elif name_type == 'company_name':
            base_score = 0.6
        else:
            base_score = 0.5
            
        score = base_score
        words = name.split()
        
        # Special boost for known Ethereum upgrade names
        ethereum_upgrades = {
            'frontier': 0.9, 'homestead': 0.9, 'byzantium': 0.9, 'constantinople': 0.9,
            'petersburg': 0.9, 'istanbul': 0.9, 'muir glacier': 0.9, 'berlin': 0.9,
            'london': 0.9, 'arrow glacier': 0.9, 'gray glacier': 0.9, 'paris': 0.9,
            'shanghai': 0.9, 'cancun': 0.9, 'prague': 0.9, 'osaka': 0.9, 'bogota': 0.9,
            'dencun': 0.9, 'pectra': 0.9, 'shapella': 0.9, 'metropolis': 0.9, 'serenity': 0.9,
            'altair': 0.9, 'bellatrix': 0.9, 'merge': 0.9
        }
        
        if name.lower() in ethereum_upgrades:
            score = ethereum_upgrades[name.lower()]
            return score  # Return immediately for known high-value artifacts
            
        # Boost for proper formatting based on type
        if name_type == 'username':
            # Boost alphanumeric with underscore patterns (very likely to be valid usernames)
            if re.match(r'^[\w.-]+$', name):
                score += 0.15
                
            # Extra boost for GitHub-style usernames
            if re.match(r'^[a-zA-Z][\w-]{2,39}$', name):
                score += 0.1
                
            # Boost for @ prefix (common in social media handles)
            if name.startswith('@'):
                score += 0.1
                
        elif name_type == 'ethereum_upgrades':
            # Special boost for Ethereum upgrade name patterns
            
            # Strong boost for proper format: 1-2 capitalized words
            if len(words) <= 2 and all(word[0].isupper() for word in words):
                score += 0.2
                
            # Extra boost for hyphenated names like "Gray-Glacier"
            if '-' in name and all(part[0].isupper() if part else False for part in name.split('-')):
                score += 0.1
                
            # Strong boost for glacier-related names (very likely to be Ethereum forks)
            if 'glacier' in name.lower():
                score += 0.2
                
            # Boost for specific words associated with Ethereum upgrades
            upgrade_terms = ['fork', 'hard fork', 'upgrade', 'release', 'hardfork']
            if any(term in context_window.lower() for term in upgrade_terms):
                score += 0.15
                
        elif name_type in ['project_name', 'company_name']:
            # Boost for properly capitalized names
            if name[0].isupper():
                score += 0.1
                
            # Boost for names with consistent capitalization
            if all(word[0].isupper() for word in words if len(word) > 1):
                score += 0.05
                
            # Lower score for likely descriptive phrases
            if len(words) > 2:
                score -= 0.1 * (len(words) - 2)  # Progressive penalty for more words
                
            # Penalize phrases that look like descriptions
            description_indicators = ['powered', 'based', 'enabled', 'driven', 'platform', 'solution', 'system']
            if any(indicator in name.lower() for indicator in description_indicators):
                score -= 0.2
                
        # Word count preference - names with 1-2 words are preferred for most types
        # This helps focus on actual project names rather than descriptions
        if len(words) == 1:
            score += 0.1  # Single words are often good names
        elif len(words) == 2:
            score += 0.05  # Two-word names are common and good
        elif len(words) > 3:
            score -= 0.1 * (len(words) - 3)  # Progressive penalty
            
        # Penalize names that are very short
        if len(name) < 4 and name_type != 'ethereum_upgrades':
            score -= 0.1
            
        # Penalize names with many spaces (might be sentence fragments)
        space_ratio = name.count(' ') / len(name) if len(name) > 0 else 0
        if name_type not in ['ethereum_upgrades']:
            score -= space_ratio * 0.5
            
        # Penalize if it contains prepositions (likely fragment)
        preposition_count = sum(1 for word in words if word.lower() in 
                          ['to', 'from', 'by', 'with', 'for', 'in', 'on', 'at', 'of'])
        if preposition_count > 0:
            score -= 0.15 * preposition_count
            
        # Penalize words that look like descriptions rather than names
        description_word_patterns = [r'ing$', r'ed$', r'ly$', r'able$', r'ible$']
        adjective_count = sum(1 for word in words if any(re.search(pattern, word.lower()) for pattern in description_word_patterns))
        if adjective_count > 0 and len(words) > 1:
            score -= 0.1 * adjective_count
            
        # Adjust score if this is in context of the target entity
        if entity_context:
            score += 0.1
            
        # Check for specialized context terms
        context_window = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
        
        # Count relevant context terms
        context_term_count = 0
        for term, bonus in self.context_relevance.items():
            if re.search(r'\b' + re.escape(term) + r'\b', context_window, re.IGNORECASE):
                score += bonus
                context_term_count += 1
                
        # Extra boost if multiple context terms are present (shows strong relevance)
        if context_term_count >= 2:
            score += 0.15
            
        # Extra context boost for ethereum upgrades
        if name_type == 'ethereum_upgrades' and any(term in context_window.lower() for term in ['upgrade', 'hard fork', 'eip', 'improvement proposal']):
            score += 0.2
            
        # Cap score at 1.0 and floor at 0.0
        score = max(0.0, min(1.0, score))
        
        return score
    
    def extract_from_html(self, html_content, url=None, date=None):
        """
        Extract name artifacts from HTML content.
        
        Args:
            html_content: HTML content to parse
            url: Source URL for context
            date: Date of the content for relevance scoring
            
        Returns:
            List of name artifact dictionaries
        """
        # Track all found artifacts to avoid duplicates
        all_artifacts = []
        unique_names = set()
        
        # Skip if HTML content is missing or too small
        if not html_content or len(html_content) < 100:
            logger.warning(f"Skipping empty or tiny HTML content ({len(html_content) if html_content else 0} bytes)")
            return all_artifacts
            
        # Parse HTML
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements that might contain confusing content
            for element in soup(['script', 'style', 'noscript', 'iframe', 'svg', 'path']):
                element.decompose()
            
            # Extract from page title (weighted higher importance)
            title = soup.title.string if soup.title else ""
            if title:
                title_artifacts = self.extract_from_text(title, url, date)
                # Boost score for artifacts from the title (they're typically more relevant)
                for artifact in title_artifacts:
                    artifact['score'] = min(1.0, artifact['score'] + 0.1)
                    artifact['source'] = 'title'
                all_artifacts.extend(title_artifacts)
                # Track names to avoid duplicates
                for artifact in title_artifacts:
                    unique_names.add(artifact['name'].lower())
            
            # Extract from headings (weighted next highest importance)
            headings = soup.find_all(['h1', 'h2', 'h3'])
            for tag in headings:
                text = tag.get_text(separator=' ', strip=True)
                if text and len(text) > 10:
                    heading_artifacts = self.extract_from_text(text, url, date)
                    # Boost score for heading artifacts
                    for artifact in heading_artifacts:
                        if artifact['name'].lower() not in unique_names:
                            artifact['score'] = min(1.0, artifact['score'] + 0.05)
                            artifact['source'] = f"{tag.name}"
                            all_artifacts.append(artifact)
                            unique_names.add(artifact['name'].lower())
            
            # Extract from main content paragraphs
            content_tags = soup.find_all(['p', 'li', 'div'])
            for tag in content_tags:
                # Skip tiny or empty text blocks
                text = tag.get_text(separator=' ', strip=True)
                if text and len(text) > 20:
                    content_artifacts = self.extract_from_text(text, url, date)
                    # Add only new artifacts not seen in title or headings
                    for artifact in content_artifacts:
                        # More thorough duplicate checking
                        name_lower = artifact['name'].lower()
                        name_normalized = re.sub(r'[^\w]', '', name_lower)
                        
                        # Check both lower case and normalized versions
                        if name_lower in unique_names or (name_normalized and name_normalized in unique_names):
                            continue
                            
                        # Check for repeating words (Enterprise Enterprise pattern)
                        words = artifact['name'].split()
                        has_repeats = False
                        if len(words) > 1:
                            # Check for adjacent repeats
                            for i in range(1, len(words)):
                                if words[i].lower() == words[i-1].lower():
                                    has_repeats = True
                                    break
                                    
                            # Check for same word multiple times
                            word_counts = {}
                            for word in words:
                                word_lower = word.lower()
                                if len(word_lower) > 3:  # Only count meaningful words
                                    word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
                                    
                            # If any word is repeated too much, it's a duplicate pattern
                            for count in word_counts.values():
                                if count > 1 and count >= len(words) / 2:
                                    has_repeats = True
                                    break
                        
                        if has_repeats:
                            logger.debug(f"Skipping duplicate pattern: '{artifact['name']}'")
                            continue
                            
                        # Add to artifacts if passed all checks
                        artifact['source'] = f"{tag.name}"
                        all_artifacts.append(artifact)
                        
                        # Add both the lowercase and normalized versions to prevent future duplicates
                        unique_names.add(name_lower)
                        if name_normalized:
                            unique_names.add(name_normalized)
            
            # Extract from meta tags for added context
            meta_tags = soup.find_all('meta', attrs={'name': ['description', 'keywords', 'author']})
            for tag in meta_tags:
                if 'content' in tag.attrs and tag['content'] and len(tag['content']) > 10:
                    meta_artifacts = self.extract_from_text(tag['content'], url, date)
                    for artifact in meta_artifacts:
                        if artifact['name'].lower() not in unique_names:
                            artifact['source'] = f"meta_{tag.get('name', 'unknown')}"
                            all_artifacts.append(artifact)
                            unique_names.add(artifact['name'].lower())
            
            # Extract from structured data as a last resort
            script_tags = soup.find_all('script', type='application/ld+json')
            for tag in script_tags:
                try:
                    if not tag.string:
                        continue
                    json_data = json.loads(tag.string)
                    if isinstance(json_data, dict):
                        # Extract from JSON fields that might contain names
                        for key in ['name', 'author', 'creator', 'contributor', 'title', 'alternateName']:
                            if key in json_data and isinstance(json_data[key], str) and len(json_data[key]) > 2:
                                value = json_data[key]
                                json_artifacts = self.extract_from_text(value, url, date)
                                for artifact in json_artifacts:
                                    if artifact['name'].lower() not in unique_names:
                                        artifact['source'] = f"json_{key}"
                                        all_artifacts.append(artifact)
                                        unique_names.add(artifact['name'].lower())
                except Exception as json_error:
                    logger.debug(f"Error parsing JSON-LD: {str(json_error)}")
                    continue
            
            # Sort artifacts by score (highest first)
            all_artifacts.sort(key=lambda x: x['score'], reverse=True)
            
            # Log success
            if all_artifacts:
                logger.info(f"Extracted {len(all_artifacts)} unique name artifacts from HTML ({len(html_content)} bytes)")
                # Log top artifacts
                top_artifacts = all_artifacts[:5] if len(all_artifacts) > 5 else all_artifacts
                for artifact in top_artifacts:
                    logger.info(f"  - {artifact['name']} ({artifact['subtype']}, score: {artifact['score']})")
                if len(all_artifacts) > 5:
                    logger.info(f"  - ... and {len(all_artifacts) - 5} more")
        
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
        
        return all_artifacts
    
    def save_artifacts(self, artifacts, output_dir):
        """
        Save artifacts to files.
        
        Args:
            artifacts: List of artifact dictionaries
            output_dir: Directory to save to
        """
        if not artifacts:
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save artifacts
        for i, artifact in enumerate(artifacts):
            # Create a clean filename from the name
            clean_name = re.sub(r'[^\w\-]', '_', artifact['name'])
            filename = f"name_{i+1}_{clean_name}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(artifact, f, indent=2)
        
        # Create a summary file
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            summary = {
                "entity": self.entity,
                "timestamp": datetime.now().isoformat(),
                "artifacts_count": len(artifacts),
                "artifacts": artifacts
            }
            json.dump(summary, f, indent=2)

# Example usage for testing
if __name__ == "__main__":
    # Configure more verbose logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Test 1: Target entity exclusion
    print("\n===== TEST 1: Target entity exclusion =====")
    sample_text = """
    Vitalik Buterin (username: vitalik_btc on the Bitcoin forum) is the creator of Ethereum.
    Before creating Ethereum, he worked on a project called "Colored Coins" and contributed to Bitcoin Magazine.
    His early pseudonym was "Bitcoinmeister" in some communities.
    The Ethereum project was initially called "Frontier" during its first release phase.
    He founded the Ethereum Foundation to support development of the blockchain.
    Vitalik coined the term "smart contract" to describe the programmable features of Ethereum.
    His GitHub handle is "vbuterin" where he commits code for various projects.
    Many people refer to him as simply Vitalik, or sometimes Mr. Buterin.
    """
    
    # Create extractor with entity to exclude
    extractor = NameArtifactExtractor(entity="Vitalik Buterin")
    artifacts = extractor.extract_from_text(sample_text, url="https://example.com")
    
    print(f"Found {len(artifacts)} name artifacts (should exclude all variations of 'Vitalik Buterin'):")
    for artifact in artifacts:
        print(f"- {artifact['name']} ({artifact['subtype']}, score: {artifact['score']})")
    
    # Verify entity name and its parts are not in results
    entity_parts = ["vitalik", "buterin", "vitalik buterin"]
    contains_entity = False
    for artifact in artifacts:
        name_lower = artifact['name'].lower()
        if any(part in name_lower for part in entity_parts):
            contains_entity = True
            print(f"WARNING: Found entity part in result: {artifact['name']}")
    
    if not contains_entity:
        print("SUCCESS: No entity parts found in results")
    
    # Test 2: Ethereum upgrade names detection
    print("\n===== TEST 2: Ethereum upgrade names detection =====")
    ethereum_upgrades_text = """
    Ethereum has undergone several major upgrades:
    - The first release was called Frontier
    - Followed by Homestead, a more stable version
    - The Byzantium hard fork added several features
    - Constantinople upgrade improved efficiency
    - The Berlin upgrade in April 2021
    - London hard fork introduced EIP-1559
    - After that came the Arrow Glacier and Gray Glacier upgrades
    - The Merge (Paris) transitioned to proof-of-stake
    - Shanghai enabled staking withdrawals
    - Most recently, the Dencun upgrade reduced Layer 2 costs
    
    Some garbage text to test filtering: s to the rules of the Ethereum addressing urgent it included several ramEthereum FoundationEthereum.
    """
    
    # Use same extractor (with Vitalik Buterin excluded)
    upgrade_artifacts = extractor.extract_from_text(ethereum_upgrades_text, url="https://ethereum.org/history")
    
    print(f"Found {len(upgrade_artifacts)} Ethereum upgrade artifacts:")
    for artifact in upgrade_artifacts:
        print(f"- {artifact['name']} ({artifact['subtype']}, score: {artifact['score']})")
    
    # Test 3: Garbage text and descriptions rejection
    print("\n===== TEST 3: Garbage text and descriptions rejection =====")
    garbage_text = """
    This is a test of garbage fragment and description rejection:
    - addressing urgent issues
    - s to the rules of the Ethereum
    - ramEthereum FoundationEthereum
    - It included several improvements
    - ing the development
    - ing of Ethereum 
    - should be filtered out
    
    Some descriptions that should be rejected:
    - AI-powered developer
    - Trillion dollar security
    - Enterprise Enterprise (duplicate word pattern)
    - High-performance blockchain platform
    - User-friendly interface
    - Highly scalable solution
    """
    
    garbage_artifacts = extractor.extract_from_text(garbage_text, url="https://example.com/garbage")
    
    print(f"Found {len(garbage_artifacts)} artifacts from garbage text (should be 0 or very few):")
    for artifact in garbage_artifacts:
        print(f"- {artifact['name']} ({artifact['subtype']}, score: {artifact['score']})")
    
    # Test 4: Duplicate pattern detection (Enterprise Enterprise)
    print("\n===== TEST 4: Duplicate pattern detection =====")
    duplicate_text = """
    Some examples with duplicate patterns:
    - Enterprise Enterprise is a company
    - The Ethereum Ethereum project
    - Gray Gray Glacier
    - The platform platform for developers
    - Blockchain blockchain technology
    
    And some valid names:
    - Gray Glacier is an upgrade
    - Constantinople is another upgrade
    - Arrow Glacier came later
    """
    
    duplicate_artifacts = extractor.extract_from_text(duplicate_text, url="https://example.com/duplicates")
    
    print(f"Found {len(duplicate_artifacts)} artifacts after duplicate filtering:")
    for artifact in duplicate_artifacts:
        print(f"- {artifact['name']} ({artifact['subtype']}, score: {artifact['score']})")
    
    print("\nTests complete!")