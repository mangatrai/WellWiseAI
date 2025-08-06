#!/usr/bin/env python3
"""
Test script for database insertion functionality
"""

import os
import json
import logging
from db.insert_data import WellWiseDBInserter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_record():
    """Create a sample test record for validation."""
    return {
        "well_id": "TEST-WELL-001",
        "record_type": "logging",
        "depth_start": 1000.0,
        "curve_name": "GR",
        "field_name": "TEST_FIELD",
        "country": "NORWAY",
        "state_province": "",
        "service_company": "TEST_SERVICE",
        "acquisition_date": "2024-01-15T10:30:00.000000",
        "latitude": 58.441584,
        "longitude": 1.887541,
        "elevation": 54.9,
        "sample_mean": 45.2,
        "sample_min": 12.5,
        "sample_max": 78.9,
        "num_samples": 1000,
        "remarks": "Test record for database insertion validation"
    }

def test_single_record_insertion():
    """Test inserting a single record."""
    print("=== TESTING SINGLE RECORD INSERTION ===")
    
    # Get credentials from environment variables
    application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not application_token or not api_endpoint:
        print("❌ Please set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT environment variables")
        return False
    
    try:
        # Initialize inserter
        inserter = WellWiseDBInserter(application_token, api_endpoint)
        
        # Create test record
        test_record = create_test_record()
        
        # Test single record insertion
        success = inserter.insert_single_record(test_record)
        
        if success:
            print("✅ Single record insertion successful")
            return True
        else:
            print("❌ Single record insertion failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during single record insertion: {e}")
        return False

def test_batch_insertion():
    """Test inserting multiple records in batch."""
    print("\n=== TESTING BATCH RECORD INSERTION ===")
    
    # Get credentials from environment variables
    application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not application_token or not api_endpoint:
        print("❌ Please set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT environment variables")
        return False
    
    try:
        # Initialize inserter
        inserter = WellWiseDBInserter(application_token, api_endpoint)
        
        # Create multiple test records
        test_records = []
        for i in range(5):
            record = create_test_record()
            record['well_id'] = f"TEST-WELL-{i:03d}"
            record['curve_name'] = f"CURVE_{i}"
            record['depth_start'] = 1000.0 + (i * 100)
            test_records.append(record)
        
        # Test batch insertion
        result = inserter.insert_many_records(test_records, batch_size=2)
        
        print(f"📊 Batch insertion results:")
        print(f"   Total Records: {result['total']}")
        print(f"   Successful: {result['successful']}")
        print(f"   Failed: {result['failed']}")
        
        if result['successful'] > 0:
            print("✅ Batch insertion successful")
            return True
        else:
            print("❌ Batch insertion failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during batch insertion: {e}")
        return False

def test_json_file_insertion():
    """Test inserting from a JSON file."""
    print("\n=== TESTING JSON FILE INSERTION ===")
    
    # Get credentials from environment variables
    application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not application_token or not api_endpoint:
        print("❌ Please set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT environment variables")
        return False
    
    try:
        # Initialize inserter
        inserter = WellWiseDBInserter(application_token, api_endpoint)
        
        # Create a test JSON file
        test_records = []
        for i in range(3):
            record = create_test_record()
            record['well_id'] = f"JSON-TEST-{i:03d}"
            record['curve_name'] = f"JSON_CURVE_{i}"
            record['depth_start'] = 2000.0 + (i * 200)
            test_records.append(record)
        
        test_data = {
            "metadata": {
                "source": "test_insertion.py",
                "created_at": "2024-01-15T10:30:00.000000"
            },
            "records": test_records
        }
        
        # Write test JSON file
        test_json_path = "db/test_data.json"
        with open(test_json_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"📄 Created test JSON file: {test_json_path}")
        
        # Test JSON file insertion
        result = inserter.insert_from_json_file(test_json_path, batch_size=2)
        
        print(f"📊 JSON file insertion results:")
        print(f"   Total Records: {result['total']}")
        print(f"   Successful: {result['successful']}")
        print(f"   Failed: {result['failed']}")
        
        # Clean up test file
        if os.path.exists(test_json_path):
            os.remove(test_json_path)
            print(f"🗑️  Cleaned up test file: {test_json_path}")
        
        if result['successful'] > 0:
            print("✅ JSON file insertion successful")
            return True
        else:
            print("❌ JSON file insertion failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during JSON file insertion: {e}")
        return False

def test_data_type_conversion():
    """Test data type conversion functionality."""
    print("\n=== TESTING DATA TYPE CONVERSION ===")
    
    # Get credentials from environment variables
    application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not application_token or not api_endpoint:
        print("❌ Please set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT environment variables")
        return False
    
    try:
        # Initialize inserter
        inserter = WellWiseDBInserter(application_token, api_endpoint)
        
        # Test record with various data types
        test_record = {
            "well_id": "TYPE-TEST-001",
            "record_type": "logging",
            "depth_start": 1500.0,
            "curve_name": "TYPE_TEST",
            "field_name": "TEST_FIELD",
            "country": "NORWAY",
            "acquisition_date": "2024-01-15T10:30:00.000000",
            "latitude": 58.441584,
            "longitude": 1.887541,
            "elevation": 54.9,
            "sample_mean": 45.2,
            "sample_min": 12.5,
            "sample_max": 78.9,
            "num_samples": 1000,
            "remarks": "Test record with various data types for conversion validation",
            "boolean_field": True,
            "null_field": None,
            "empty_field": ""
        }
        
        # Test data type conversion
        converted_record = inserter.convert_data_types(test_record)
        
        print("📋 Original record keys:", list(test_record.keys()))
        print("📋 Converted record keys:", list(converted_record.keys()))
        
        # Check that null/empty values are filtered out
        if 'null_field' not in converted_record and 'empty_field' not in converted_record:
            print("✅ Null/empty value filtering working correctly")
        else:
            print("❌ Null/empty value filtering not working")
            return False
        
        # Check that date conversion works
        if 'acquisition_date' in converted_record:
            print("✅ Date conversion working correctly")
        else:
            print("❌ Date conversion not working")
            return False
        
        print("✅ Data type conversion successful")
        return True
        
    except Exception as e:
        print(f"❌ Error during data type conversion: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting database insertion tests...\n")
    
    tests = [
        ("Data Type Conversion", test_data_type_conversion),
        ("Single Record Insertion", test_single_record_insertion),
        ("Batch Record Insertion", test_batch_insertion),
        ("JSON File Insertion", test_json_file_insertion)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED\n")
            else:
                print(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}\n")
    
    print("=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Database insertion is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the configuration and try again.")

if __name__ == "__main__":
    main() 