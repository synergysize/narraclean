"""
Fix for the name artifact discovery issue in enhanced_artifact_detector.py

This patch fixes the field mapping issue where name artifacts were being extracted
but not properly added to discoveries because the 'name' field was not being mapped
to the 'content' field expected by the detective agent.

Issue:
- NameArtifactExtractor stores the name value in the 'name' field
- DetectiveAgent._process_artifacts expects the value to be in the 'content' field
- This mismatch prevents name artifacts from being properly recorded as discoveries

Fix:
- Update the _extract_name_artifacts method to map the 'name' field to 'content'
"""

# ORIGINAL CODE (enhanced_artifact_detector.py, lines 111-123):
"""
standardized = {
    "type": "name",
    "subtype": artifact.get("subtype", "unknown"),
    "content": artifact.get("context", ""),  # THIS IS THE ISSUE - using context instead of name
    "summary": f"Name artifact: {artifact.get('name', '')} ({artifact.get('subtype', 'unknown')})",
    "location": "HTML content",
    "hash": "",  # Would normally generate a hash
    "score": artifact.get("score", 0.5),
    "url": url,
    "date": date,
    "entity": entity,
    "name": artifact.get("name", "")
}
"""

# FIXED CODE:
"""
standardized = {
    "type": "name",
    "subtype": artifact.get("subtype", "unknown"),
    "content": artifact.get("name", ""),  # FIXED - now uses name instead of context
    "summary": f"Name artifact: {artifact.get('name', '')} ({artifact.get('subtype', 'unknown')})",
    "location": "HTML content",
    "hash": "",  # Would normally generate a hash
    "score": artifact.get("score", 0.5),
    "url": url,
    "date": date,
    "entity": entity,
    "name": artifact.get("name", "")
}
"""

# Testing the fix:
# 1. Make the change in enhanced_artifact_detector.py
# 2. Run the detective agent with the same objective
# 3. Verify that name artifacts are properly added to discoveries
# 4. The list of discoveries should now include all high-confidence name artifacts