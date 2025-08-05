#!/usr/bin/env python3
"""
Test script for DLIS parser functionality.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from parsers.dlis import parse_dlis

def setup_test_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('test_dlis')

def test_dlis_parser():
    """Test the DLIS parser with sample data."""
    logger = setup_test_logging()
    
    # Test with a sample DLIS file path
    test_file = "sample.dlis"  # This would be a real DLIS file
    
    print("Testing DLIS parser...")
    print(f"Looking for test file: {test_file}")
    
    if os.path.exists(test_file):
        print(f"Found test file: {test_file}")
        
        try:
            # Parse the DLIS file
            records = parse_dlis(test_file, logger)
            
            print(f"Successfully parsed {len(records)} records")
            
            if records:
                print("\nSample record structure:")
                sample_record = records[0]
                for key, value in sample_record.items():
                    print(f"  {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"Error parsing DLIS file: {e}")
            return False
    else:
        print(f"Test file not found: {test_file}")
        print("Please place a sample DLIS file in the project directory for testing.")
        return False

if __name__ == "__main__":
    success = test_dlis_parser()
    sys.exit(0 if success else 1) 