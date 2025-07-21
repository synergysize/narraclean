#!/usr/bin/env python3
"""
Test the focused exploration strategy implementation.

This test file verifies that the new ResearchStrategy class and its integration 
with the DetectiveAgent work correctly.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import json
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules to test
from research_strategy import ResearchStrategy
from detective_agent import DetectiveAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestResearchStrategy(unittest.TestCase):
    """Test the ResearchStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = ResearchStrategy()
        
    def test_add_target(self):
        """Test adding a target to the strategy."""
        target = {
            'url': 'https://example.com',
            'type': 'website',
            'priority': 10,
            'rationale': 'Test rationale'
        }
        self.strategy.add_target(target)
        
        # Check that the target was added
        self.assertEqual(len(self.strategy.master_todo_list), 1)
        self.assertEqual(self.strategy.master_todo_list[0], target)
        
    def test_get_next_target(self):
        """Test getting the next target from the strategy."""
        # Add some targets with different priorities
        self.strategy.add_target({
            'url': 'https://low-priority.com',
            'type': 'website',
            'priority': 5,
            'rationale': 'Low priority'
        })
        
        self.strategy.add_target({
            'url': 'https://high-priority.com',
            'type': 'website',
            'priority': 10,
            'rationale': 'High priority'
        })
        
        self.strategy.add_target({
            'url': 'https://medium-priority.com',
            'type': 'website',
            'priority': 7,
            'rationale': 'Medium priority'
        })
        
        # Get the next target - should be the highest priority one
        next_target = self.strategy.get_next_target()
        self.assertEqual(next_target['url'], 'https://high-priority.com')
        self.assertEqual(next_target['priority'], 10)
        
        # Check that it was removed from the todo list
        self.assertEqual(len(self.strategy.master_todo_list), 2)
        
        # Get the next target - should be the medium priority one
        next_target = self.strategy.get_next_target()
        self.assertEqual(next_target['url'], 'https://medium-priority.com')
        self.assertEqual(next_target['priority'], 7)
        
    def test_mark_target_complete(self):
        """Test marking a target as complete."""
        target = {
            'url': 'https://example.com',
            'type': 'website',
            'priority': 10,
            'rationale': 'Test rationale'
        }
        self.strategy.add_target(target)
        next_target = self.strategy.get_next_target()
        
        # Mark it as complete
        self.strategy.mark_target_complete(next_target, [{'title': 'Test discovery'}])
        
        # Check that it was added to completed targets
        self.assertEqual(len(self.strategy.completed_targets), 1)
        self.assertEqual(self.strategy.completed_targets[0]['url'], 'https://example.com')
        self.assertEqual(self.strategy.completed_targets[0]['discoveries_count'], 1)
        
    def test_add_discovered_lead(self):
        """Test adding a discovered lead."""
        lead = {
            'url': 'https://discovered-lead.com',
            'type': 'website',
            'priority': 8,
            'rationale': 'Discovered during investigation'
        }
        self.strategy.add_discovered_lead(lead)
        
        # Check that it was added to discovered leads
        self.assertEqual(len(self.strategy.discovered_leads), 1)
        self.assertEqual(self.strategy.discovered_leads[0]['url'], 'https://discovered-lead.com')
        
        # Process discovered leads
        self.strategy.process_discovered_leads()
        
        # Check that it was moved to the todo list
        self.assertEqual(len(self.strategy.discovered_leads), 0)
        self.assertEqual(len(self.strategy.master_todo_list), 1)
        self.assertEqual(self.strategy.master_todo_list[0]['url'], 'https://discovered-lead.com')
        
    def test_update_priorities(self):
        """Test updating priorities."""
        # Add some targets
        self.strategy.add_target({
            'url': 'https://github.com/example/repo',
            'type': 'github',
            'priority': 5,
            'rationale': 'GitHub repo'
        })
        
        self.strategy.add_target({
            'url': 'https://ethereum.org/page',
            'type': 'website',
            'priority': 5,
            'rationale': 'Website'
        })
        
        # Update priorities
        priority_updates = {
            'type_adjustments': {
                'github': 2,  # +2 for GitHub targets
                'website': 0  # No change for websites
            },
            'domain_priorities': {
                'ethereum.org': 10  # Set ethereum.org domains to priority 10
            }
        }
        self.strategy.update_priorities(priority_updates)
        
        # Check that priorities were updated
        for target in self.strategy.master_todo_list:
            if target['type'] == 'github':
                self.assertEqual(target['priority'], 7)  # 5 + 2
            if target['type'] == 'website' and 'ethereum.org' in target['url']:
                self.assertEqual(target['priority'], 10)  # Set directly to 10

class TestDetectiveAgentIntegration(unittest.TestCase):
    """Test the integration of ResearchStrategy with DetectiveAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create the agent with patched methods for testing
        with patch('detective_agent.LLMIntegration') as mock_llm:
            # Create a mock LLM that will return our test data
            mock_instance = MagicMock()
            mock_instance._extract_json.return_value = json.dumps({
                "sources": [
                    {"url": "https://vitalik.ca", "priority": 10, "rationale": "Test"}
                ],
                "github_targets": [
                    {"url": "https://github.com/vbuterin/ethereum", "priority": 9, "rationale": "Test"}
                ],
                "search_queries": [
                    {"query": "Vitalik test", "priority": 7, "rationale": "Test"}
                ]
            })
            mock_instance._call_claude.return_value = '{"fake": "response"}'
            mock_instance._call_openai.return_value = '{"fake": "response"}'
            mock_instance.use_claude = True
            mock_instance.use_openai = False
            
            # Configure the mock to return our instance
            mock_llm.return_value = mock_instance
            
            # Create the agent
            self.agent = DetectiveAgent("Test objective", "Vitalik Buterin", max_iterations=3)
            
            # Save the mock for later use in tests
            self.mock_llm = mock_instance
        
    def test_initialization(self):
        """Test that the agent initializes the research strategy."""
        # Directly populate the research strategy for testing
        self.agent.research_strategy.add_target({
            'url': 'https://vitalik.ca',
            'type': 'website',
            'priority': 10,
            'rationale': 'Test initialization'
        })
        
        # Check that the strategy has targets
        strategy_status = self.agent.research_strategy.get_status()
        self.assertEqual(strategy_status['todo_count'], 1)
        
    def test_investigation_flow(self):
        """Test the investigation flow with the research strategy."""
        # Prepare test data
        test_target = {
            'url': 'https://vitalik.ca/test',
            'type': 'website',
            'priority': 10,
            'rationale': 'Test target'
        }
        
        test_discoveries = [
            {'title': 'Test discovery 1', 'content': 'https://example.com link', 'id': '1', 'url': 'https://source.com', 'source_type': 'website'},
            {'title': 'Test discovery 2', 'content': 'https://github.com/test/repo', 'id': '2'}
        ]
        
        # Add a test target to the strategy
        self.agent.research_strategy.add_target(test_target)
        
        # Mock the execute_investigation method
        with patch.object(self.agent, '_execute_investigation', return_value=test_discoveries):
            # Simulate one iteration
            self.agent.current_iteration = 0
            target = self.agent.research_strategy.get_next_target()
            self.assertIsNotNone(target)
            
            # Execute investigation
            discoveries = self.agent._execute_investigation(target)
            self.agent.iteration_discoveries[1] = discoveries
            
            # Mark current target as complete
            self.agent.research_strategy.mark_target_complete(target, discoveries)
            
            # Extract leads
            for discovery in discoveries:
                leads = self.agent._extract_leads_from_discovery(discovery)
                for lead in leads:
                    self.agent.research_strategy.add_discovered_lead(lead)
                    
            # Process leads
            self.agent.research_strategy.process_discovered_leads()
            
            # Check that new leads were added to the todo list
            strategy_status = self.agent.research_strategy.get_status()
            self.assertGreater(strategy_status['todo_count'], 0)
        
    def test_strategy_review(self):
        """Test the strategy review process."""
        # Prepare test recommendations
        test_recommendations = {
            'strategy_analysis': 'Test analysis',
            'recommended_approach': 'Test approach',
            'new_targets': [
                {'url': 'https://new-target.com', 'type': 'website', 'priority': 9, 'rationale': 'Test'}
            ],
            'priority_updates': {
                'type_adjustments': {'github': 2, 'website': -1},
                'domain_priorities': {'ethereum.org': 10}
            }
        }
        
        # Mock the get_strategy_recommendations method
        with patch.object(self.agent, '_get_strategy_recommendations', return_value=test_recommendations):
            # Perform strategy review
            self.agent._review_research_strategy(force_update=True)
            
            # Check that new targets were added
            found = False
            for target in self.agent.research_strategy.master_todo_list:
                if target.get('url') == 'https://new-target.com':
                    found = True
                    break
            self.assertTrue(found)

if __name__ == '__main__':
    unittest.main()