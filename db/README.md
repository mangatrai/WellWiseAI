# Database Insertion Module

This module provides comprehensive database insertion functionality for WellWiseAI using Astra DB's Data API.

## Overview

The database insertion module consists of:

- **`insert_data.py`**: Main insertion class with batch processing capabilities
- **`test_insertion.py`**: Test suite for validation
- **`README.md`**: This documentation

## Features

### 🚀 Core Functionality

- **Batch Processing**: Insert multiple records efficiently
- **Data Type Conversion**: Automatic conversion to Astra DB compatible types
- **Primary Key Validation**: Ensures all required fields are present
- **Error Handling**: Comprehensive error handling and logging
- **Progress Tracking**: Real-time progress reporting

### 📊 Data Type Support

- **Dates**: Automatic conversion to `DataAPIDate`
- **Numbers**: Integers and floats
- **Strings**: Text fields with length validation
- **Booleans**: True/false values
- **Null Handling**: Automatic filtering of null/empty values

## Setup

### 1. Environment Variables

Set the following environment variables:

```bash
export ASTRA_DB_APPLICATION_TOKEN="your_application_token"
export ASTRA_DB_API_ENDPOINT="your_api_endpoint"
```

### 2. Install Dependencies

```bash
uv pip install astrapy
```

## Usage

### Basic Usage

```python
from db.insert_data import WellWiseDBInserter

# Initialize inserter
inserter = WellWiseDBInserter(application_token, api_endpoint)

# Insert single record
success = inserter.insert_single_record(record)

# Insert multiple records
result = inserter.insert_many_records(records, batch_size=100)

# Insert from JSON file
result = inserter.insert_from_json_file("parsed_data/file.json")

# Insert all data from directory
result = inserter.insert_from_parsed_data_directory()
```

### Complete Pipeline

Use the main pipeline script for end-to-end processing:

```bash
# Run complete pipeline (parse + insert)
python wellwise_pipeline.py

# Skip parsing (data already exists)
python wellwise_pipeline.py --skip-parsing

# Skip insertion (only parse)
python wellwise_pipeline.py --skip-insertion
```

## Testing

### Run Test Suite

```bash
python db/test_insertion.py
```

The test suite validates:

- ✅ Data type conversion
- ✅ Single record insertion
- ✅ Batch record insertion
- ✅ JSON file insertion

### Test Results

```
🚀 Starting database insertion tests...

=== TESTING DATA TYPE CONVERSION ===
✅ Data type conversion successful

=== TESTING SINGLE RECORD INSERTION ===
✅ Single record insertion successful

=== TESTING BATCH RECORD INSERTION ===
📊 Batch insertion results:
   Total Records: 5
   Successful: 5
   Failed: 0
✅ Batch insertion successful

=== TESTING JSON FILE INSERTION ===
📄 Created test JSON file: db/test_data.json
📊 JSON file insertion results:
   Total Records: 3
   Successful: 3
   Failed: 0
✅ JSON file insertion successful

==================================================
📊 TEST RESULTS: 4/4 tests passed
🎉 All tests passed! Database insertion is working correctly.
```

## Data Schema

### Primary Key Fields

The following fields are required for all records:

- `well_id`: Well identifier
- `record_type`: Type of record (e.g., "logging")
- `depth_start`: Starting depth
- `curve_name`: Curve/measurement name

### Supported Fields

All 75 canonical fields from the schema are supported:

- **Geographic**: `latitude`, `longitude`, `elevation`, `country`, `state_province`
- **Well Information**: `field_name`, `service_company`, `acquisition_date`
- **Petrophysical**: `porosity`, `permeability`, `resistivity_deep`, etc.
- **Statistics**: `sample_mean`, `sample_min`, `sample_max`, `num_samples`
- **Metadata**: `remarks`, `file_origin`, `version`

## Error Handling

### Common Issues

1. **Missing Primary Key Fields**
   ```
   ❌ Missing primary key field: well_id
   ```

2. **Invalid Data Types**
   ```
   ⚠️ Could not convert date acquisition_date: invalid format
   ```

3. **Database Connection Issues**
   ```
   ❌ Error inserting record: Connection timeout
   ```

### Troubleshooting

1. **Check Environment Variables**
   ```bash
   echo $ASTRA_DB_APPLICATION_TOKEN
   echo $ASTRA_DB_API_ENDPOINT
   ```

2. **Validate Data Format**
   ```python
   # Test data type conversion
   converted = inserter.convert_data_types(record)
   print(converted)
   ```

3. **Check Primary Keys**
   ```python
   # Validate record
   valid = inserter.validate_primary_key(record)
   print(f"Valid: {valid}")
   ```

## Performance

### Batch Sizing

- **Small Batches (10-50)**: Good for testing and small datasets
- **Medium Batches (100-500)**: Optimal for most use cases
- **Large Batches (1000+)**: Use with caution, monitor memory usage

### Memory Usage

- Records are processed in batches to manage memory
- Large JSON files are read incrementally
- Null/empty values are filtered to reduce payload size

## Monitoring

### Logging Levels

- **INFO**: General progress and results
- **DEBUG**: Detailed operation information
- **WARNING**: Non-critical issues
- **ERROR**: Critical failures

### Metrics

The insertion process provides detailed metrics:

- Total records processed
- Successful insertions
- Failed insertions
- Success rate percentage
- Processing duration

## Integration

### With Parsing Pipeline

The database insertion integrates seamlessly with the parsing pipeline:

```python
from wellwise_pipeline import WellWisePipeline

pipeline = WellWisePipeline()
results = pipeline.run_complete_pipeline()
```

### With Custom Data

```python
# Custom record insertion
custom_record = {
    "well_id": "CUSTOM-WELL-001",
    "record_type": "logging",
    "depth_start": 1000.0,
    "curve_name": "CUSTOM_CURVE",
    "field_name": "CUSTOM_FIELD",
    # ... other fields
}

inserter.insert_single_record(custom_record)
```

## Security

### Best Practices

1. **Environment Variables**: Store credentials securely
2. **Token Rotation**: Regularly rotate application tokens
3. **Access Control**: Use minimal required permissions
4. **Data Validation**: Validate all input data

### Network Security

- Uses HTTPS for all API communications
- Supports connection timeouts and retries
- Validates SSL certificates

## Support

For issues and questions:

1. Check the test suite: `python db/test_insertion.py`
2. Review logs for detailed error messages
3. Validate environment variables and credentials
4. Test with a single record first

## References

- [Astra DB Data API Documentation](https://docs.datastax.com/en/astra-db-serverless/api-reference/row-methods/insert-many.html)
- [Data Types Reference](https://docs.datastax.com/en/astra-db-serverless/api-reference/table-data-types.html)
- [WellWiseAI Main Documentation](../README.md) 