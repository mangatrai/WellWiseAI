#!/usr/bin/env python3
"""
Example usage of the WellWiseAI database insertion functionality
"""

import os
import json
from db.insert_data import WellWiseDBInserter

def example_single_record():
    """Example: Insert a single record."""
    print("=== EXAMPLE: Single Record Insertion ===")
    
    # Get credentials from environment
    application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not application_token or not api_endpoint:
        print("❌ Please set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT")
        return
    
    # Initialize inserter
    inserter = WellWiseDBInserter(application_token, api_endpoint)
    
    # Create a sample record
    sample_record = {
        "well_id": "EXAMPLE-WELL-001",
        "record_type": "logging",
        "depth_start": 1000.0,
        "curve_name": "GR",
        "field_name": "EXAMPLE_FIELD",
        "country": "NORWAY",
        "service_company": "EXAMPLE_SERVICE",
        "acquisition_date": "2024-01-15T10:30:00.000000",
        "latitude": 58.441584,
        "longitude": 1.887541,
        "elevation": 54.9,
        "sample_mean": 45.2,
        "sample_min": 12.5,
        "sample_max": 78.9,
        "num_samples": 1000,
        "remarks": "Example record for demonstration"
    }
    
    # Insert the record
    success = inserter.insert_single_record(sample_record)
    
    if success:
        print("✅ Single record inserted successfully")
    else:
        print("❌ Single record insertion failed")

def example_batch_insertion():
    """Example: Insert multiple records in batch."""
    print("\n=== EXAMPLE: Batch Record Insertion ===")
    
    # Get credentials from environment
    application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not application_token or not api_endpoint:
        print("❌ Please set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT")
        return
    
    # Initialize inserter
    inserter = WellWiseDBInserter(application_token, api_endpoint)
    
    # Create multiple sample records
    sample_records = []
    for i in range(3):
        record = {
            "well_id": f"BATCH-WELL-{i:03d}",
            "record_type": "logging",
            "depth_start": 1000.0 + (i * 100),
            "curve_name": f"CURVE_{i}",
            "field_name": "BATCH_FIELD",
            "country": "NORWAY",
            "service_company": "BATCH_SERVICE",
            "acquisition_date": "2024-01-15T10:30:00.000000",
            "latitude": 58.441584,
            "longitude": 1.887541,
            "elevation": 54.9,
            "sample_mean": 45.2 + i,
            "sample_min": 12.5,
            "sample_max": 78.9,
            "num_samples": 1000,
            "remarks": f"Batch example record {i+1}"
        }
        sample_records.append(record)
    
    # Insert records in batch
    result = inserter.insert_many_records(sample_records, batch_size=2)
    
    print(f"📊 Batch insertion results:")
    print(f"   Total Records: {result['total']}")
    print(f"   Successful: {result['successful']}")
    print(f"   Failed: {result['failed']}")
    
    if result['successful'] > 0:
        print("✅ Batch insertion successful")
    else:
        print("❌ Batch insertion failed")

def example_json_file_insertion():
    """Example: Insert from a JSON file."""
    print("\n=== EXAMPLE: JSON File Insertion ===")
    
    # Get credentials from environment
    application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not application_token or not api_endpoint:
        print("❌ Please set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT")
        return
    
    # Initialize inserter
    inserter = WellWiseDBInserter(application_token, api_endpoint)
    
    # Create sample JSON file
    sample_data = {
        "metadata": {
            "source": "example_usage.py",
            "created_at": "2024-01-15T10:30:00.000000"
        },
        "records": [
            {
                "well_id": "JSON-WELL-001",
                "record_type": "logging",
                "depth_start": 1000.0,
                "curve_name": "JSON_CURVE_1",
                "field_name": "JSON_FIELD",
                "country": "NORWAY",
                "service_company": "JSON_SERVICE",
                "acquisition_date": "2024-01-15T10:30:00.000000",
                "latitude": 58.441584,
                "longitude": 1.887541,
                "elevation": 54.9,
                "sample_mean": 45.2,
                "sample_min": 12.5,
                "sample_max": 78.9,
                "num_samples": 1000,
                "remarks": "JSON file example record 1"
            },
            {
                "well_id": "JSON-WELL-002",
                "record_type": "logging",
                "depth_start": 1100.0,
                "curve_name": "JSON_CURVE_2",
                "field_name": "JSON_FIELD",
                "country": "NORWAY",
                "service_company": "JSON_SERVICE",
                "acquisition_date": "2024-01-15T10:30:00.000000",
                "latitude": 58.441584,
                "longitude": 1.887541,
                "elevation": 54.9,
                "sample_mean": 46.2,
                "sample_min": 13.5,
                "sample_max": 79.9,
                "num_samples": 1000,
                "remarks": "JSON file example record 2"
            }
        ]
    }
    
    # Write sample JSON file
    json_file_path = "db/example_data.json"
    with open(json_file_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"📄 Created sample JSON file: {json_file_path}")
    
    # Insert from JSON file
    result = inserter.insert_from_json_file(json_file_path, batch_size=2)
    
    print(f"📊 JSON file insertion results:")
    print(f"   Total Records: {result['total']}")
    print(f"   Successful: {result['successful']}")
    print(f"   Failed: {result['failed']}")
    
    # Clean up
    if os.path.exists(json_file_path):
        os.remove(json_file_path)
        print(f"🗑️  Cleaned up sample file: {json_file_path}")
    
    if result['successful'] > 0:
        print("✅ JSON file insertion successful")
    else:
        print("❌ JSON file insertion failed")

def example_data_type_conversion():
    """Example: Demonstrate data type conversion."""
    print("\n=== EXAMPLE: Data Type Conversion ===")
    
    # Get credentials from environment
    application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not application_token or not api_endpoint:
        print("❌ Please set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT")
        return
    
    # Initialize inserter
    inserter = WellWiseDBInserter(application_token, api_endpoint)
    
    # Create record with various data types
    test_record = {
        "well_id": "TYPE-TEST-001",
        "record_type": "logging",
        "depth_start": 1500.0,
        "curve_name": "TYPE_TEST",
        "field_name": "TYPE_FIELD",
        "country": "NORWAY",
        "acquisition_date": "2024-01-15T10:30:00.000000",
        "latitude": 58.441584,
        "longitude": 1.887541,
        "elevation": 54.9,
        "sample_mean": 45.2,
        "sample_min": 12.5,
        "sample_max": 78.9,
        "num_samples": 1000,
        "remarks": "Test record with various data types",
        "boolean_field": True,
        "null_field": None,
        "empty_field": ""
    }
    
    print("📋 Original record:")
    for key, value in test_record.items():
        print(f"   {key}: {value} ({type(value).__name__})")
    
    # Convert data types
    converted_record = inserter.convert_data_types(test_record)
    
    print("\n📋 Converted record:")
    for key, value in converted_record.items():
        print(f"   {key}: {value} ({type(value).__name__})")
    
    # Check what was filtered out
    original_keys = set(test_record.keys())
    converted_keys = set(converted_record.keys())
    filtered_keys = original_keys - converted_keys
    
    if filtered_keys:
        print(f"\n🗑️  Filtered out (null/empty): {filtered_keys}")
    else:
        print("\n✅ No null/empty values to filter")

def main():
    """Run all examples."""
    print("🚀 WellWiseAI Database Insertion Examples")
    print("=" * 50)
    
    # Check if credentials are available
    application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not application_token or not api_endpoint:
        print("⚠️  Database credentials not found")
        print("   Set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT")
        print("   Examples will show the structure but won't perform actual insertion")
        print()
    
    # Run examples
    examples = [
        ("Data Type Conversion", example_data_type_conversion),
        ("Single Record Insertion", example_single_record),
        ("Batch Record Insertion", example_batch_insertion),
        ("JSON File Insertion", example_json_file_insertion)
    ]
    
    for example_name, example_func in examples:
        try:
            example_func()
            print()
        except Exception as e:
            print(f"❌ Error in {example_name}: {e}")
            print()
    
    print("✅ Examples completed!")

if __name__ == "__main__":
    main() 