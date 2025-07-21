#!/usr/bin/env python3
"""
Validation script for testing BIP39 wordlist loading and seed phrase detection.
"""

import os
import sys
import logging
import importlib
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('validation')

# Test HTML with seed phrases
TEST_HTML = """
<html>
<body>
<h1>Test Seed Phrase Detection</h1>
<p>This is a valid seed phrase: abandon ability able about above absent absorb abstract absurd abuse access accident</p>
<p>This is another valid seed phrase: zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong</p>
<p>This is not a valid seed phrase: apple banana cherry dolphin elephant frog giraffe hippo iguana jaguar</p>
</body>
</html>
"""

def test_bip39_wordlist_loading():
    """Test that BIP39 wordlist loading works with proper validation."""
    logger.info("Testing BIP39 wordlist loading...")

    # Get the absolute path to the project directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get path to the wordlist
    wordlist_path = os.path.join(base_dir, 'config/wordlists/bip39.txt')
    wordlist_dir = os.path.dirname(wordlist_path)
    
    # Backup the original wordlist if it exists
    backup_path = None
    if os.path.exists(wordlist_path):
        logger.info(f"Backing up existing wordlist at {wordlist_path}")
        backup_path = wordlist_path + '.backup'
        shutil.copy2(wordlist_path, backup_path)
        os.remove(wordlist_path)
    
    # Test 1: Non-existent wordlist
    logger.info("==== Test 1: Non-existent wordlist ====")
    # Ensure the file doesn't exist
    if os.path.exists(wordlist_path):
        os.remove(wordlist_path)
    
    # Import artifact_extractor fresh to trigger BIP39 loading
    logger.info("Loading artifact_extractor with missing wordlist...")
    
    # Clear the module from cache if it's there
    if 'artifact_extractor' in sys.modules:
        del sys.modules['artifact_extractor']
    
    # Import the module
    import artifact_extractor
    
    # Check if minimal wordlist was created
    if os.path.exists(wordlist_path):
        with open(wordlist_path, 'r') as f:
            words = [word.strip() for word in f.readlines()]
        logger.info(f"Created minimal wordlist with {len(words)} words")
        if words:
            logger.info(f"First few words: {', '.join(words[:5])}")
    else:
        logger.error("Test failed: Minimal wordlist was not created")
    
    # Run extraction which should detect no seed phrases due to minimal wordlist
    logger.info("Running extraction with minimal wordlist...")
    artifacts = artifact_extractor.extract_artifacts_from_html(TEST_HTML, "https://test.com")
    seed_phrases = [a for a in artifacts if a['type'] == 'seed_phrase']
    logger.info(f"Seed phrases detected: {len(seed_phrases)}")
    
    # Test 2: Invalid wordlist (too small)
    logger.info("\n==== Test 2: Invalid wordlist (too small) ====")
    
    # Create an invalid wordlist (too small)
    with open(wordlist_path, 'w') as f:
        f.write('\n'.join([f"word{i}" for i in range(100)]))
    logger.info(f"Created invalid wordlist with 100 words (too small)")
    
    # Reload the module
    if 'artifact_extractor' in sys.modules:
        del sys.modules['artifact_extractor']
    
    import artifact_extractor
    
    # Test 3: Restore and verify original wordlist
    logger.info("\n==== Test 3: Restore original wordlist ====")
    
    # Restore the original wordlist if it existed
    if backup_path and os.path.exists(backup_path):
        shutil.copy2(backup_path, wordlist_path)
        os.remove(backup_path)
        logger.info(f"Restored original wordlist")
    
    # Clear the module from cache
    if 'artifact_extractor' in sys.modules:
        del sys.modules['artifact_extractor']
    
    # Re-import to test with restored wordlist
    import artifact_extractor
    
    # Run extraction with proper wordlist
    artifacts = artifact_extractor.extract_artifacts_from_html(TEST_HTML, "https://test.com")
    seed_phrases = [a for a in artifacts if a['type'] == 'seed_phrase']
    logger.info(f"Seed phrases detected with restored wordlist: {len(seed_phrases)}")
    
    logger.info("BIP39 validation tests completed")
    return True

if __name__ == "__main__":
    test_bip39_wordlist_loading()