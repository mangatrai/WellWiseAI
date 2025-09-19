#!/usr/bin/env python3
"""
Prompt Loader for WellWise AI

This module loads model-specific prompts from YAML file based on environment variables.
"""

import yaml
import os
import logging

logger = logging.getLogger('wellwise')

def load_prompts():
    """Load prompts from YAML file."""
    try:
        prompts_file = os.path.join(os.path.dirname(__file__), 'prompts.yaml')
        with open(prompts_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load prompts from YAML: {e}")
        raise

def get_prompt(parser_type, **kwargs):
    """
    Get model-specific prompt for the given parser type.
    
    Args:
        parser_type: Type of parser ('unstructured', 'dat', 'survey', 'xlsx')
        **kwargs: Variables to format into the prompt template
        
    Returns:
        Formatted prompt string
    """
    try:
        prompts = load_prompts()
        
        # Get model from environment
        if os.getenv('ENVIRONMENT', 'dev').lower() == 'production':
            model = os.getenv('CHAT_COMPLETION_MODEL', 'gpt-4o-mini')
        else:
            model = os.getenv('LOCAL_CHAT_COMPLETION_MODEL', 'llama3.1:8b')
        
        # Get prompt template
        template = prompts['models'][model]['prompts'][parser_type]
        
        # Format and return
        return template.format(**kwargs)
        
    except Exception as e:
        logger.error(f"Failed to get prompt for {parser_type}: {e}")
        raise
