#!/usr/bin/env python3
"""
Database insertion module for WellWiseAI using Astra DB Data API
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from astrapy import DataAPIClient
from astrapy.data_types import DataAPIDate, DataAPISet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WellWiseDBInserter:
    """Database inserter for WellWiseAI using Astra DB Data API."""
    
    def __init__(self, application_token: str, api_endpoint: str, keyspace_name: str = "wellwise", table_name: str = "petro_data"):
        """
        Initialize the database inserter.
        
        Args:
            application_token: Astra DB application token
            api_endpoint: Astra DB API endpoint
            keyspace_name: Database keyspace name
            table_name: Table name for petro data
        """
        self.application_token = application_token
        self.api_endpoint = api_endpoint
        self.keyspace_name = keyspace_name
        self.table_name = table_name
        
        # Initialize Astra DB client
        self.client = DataAPIClient(application_token)
        self.database = self.client.get_database(api_endpoint)
        self.table = self.database.get_table(table_name)
        
        logger.info(f"Initialized database connection to {keyspace_name}.{table_name}")
    
    def convert_data_types(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert record data types to Astra DB compatible types.
        
        Args:
            record: Raw record from parser
            
        Returns:
            Record with proper Astra DB data types
        """
        converted_record = {}
        
        for key, value in record.items():
            if value is None or value == "":
                # Skip null/empty values
                continue
                
            if key in ['acquisition_date', 'processing_date'] and value:
                # Convert date strings to DataAPIDate
                try:
                    if isinstance(value, str):
                        # Handle different date formats
                        if 'T' in value:
                            # ISO format: "2009-08-25T10:08:24.199000"
                            date_str = value.split('T')[0]
                        else:
                            # Simple date format
                            date_str = value
                        converted_record[key] = DataAPIDate.from_string(date_str)
                    else:
                        converted_record[key] = value
                except Exception as e:
                    logger.warning(f"Could not convert date {key}: {value}, error: {e}")
                    converted_record[key] = value
            
            elif key in ['remarks'] and value:
                # Truncate remarks if too long (Astra DB has limits)
                if len(str(value)) > 1000:
                    converted_record[key] = str(value)[:1000] + "..."
                else:
                    converted_record[key] = str(value)
            
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                # Numeric values
                converted_record[key] = value
            
            elif isinstance(value, str):
                # String values
                converted_record[key] = str(value)
            
            elif isinstance(value, bool):
                # Boolean values
                converted_record[key] = value
            
            else:
                # Default to string for unknown types
                converted_record[key] = str(value)
        
        return converted_record
    
    def validate_primary_key(self, record: Dict[str, Any]) -> bool:
        """
        Validate that all primary key fields are present.
        
        Args:
            record: Record to validate
            
        Returns:
            True if valid, False otherwise
        """
        primary_key_fields = ['well_id', 'record_type', 'depth_start', 'curve_name']
        
        for field in primary_key_fields:
            if not record.get(field):
                logger.error(f"Missing primary key field: {field}")
                return False
        
        return True
    
    def insert_single_record(self, record: Dict[str, Any]) -> bool:
        """
        Insert a single record into the database.
        
        Args:
            record: Record to insert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate primary key
            if not self.validate_primary_key(record):
                return False
            
            # Convert data types
            converted_record = self.convert_data_types(record)
            
            # Insert the record
            result = self.table.insert_one(converted_record)
            
            logger.debug(f"Successfully inserted record: {record.get('well_id')} - {record.get('curve_name')}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting record {record.get('well_id', 'unknown')}: {e}")
            return False
    
    def insert_many_records(self, records: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, int]:
        """
        Insert multiple records into the database in batches.
        
        Args:
            records: List of records to insert
            batch_size: Number of records to insert per batch
            
        Returns:
            Dictionary with success/failure counts
        """
        total_records = len(records)
        successful_inserts = 0
        failed_inserts = 0
        
        logger.info(f"Starting batch insertion of {total_records} records")
        
        # Process records in batches
        for i in range(0, total_records, batch_size):
            batch = records[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_records + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} records)")
            
            try:
                # Convert data types for all records in batch
                converted_records = []
                for record in batch:
                    if self.validate_primary_key(record):
                        converted_record = self.convert_data_types(record)
                        converted_records.append(converted_record)
                    else:
                        failed_inserts += 1
                
                if converted_records:
                    # Insert batch
                    result = self.table.insert_many(converted_records)
                    successful_inserts += len(converted_records)
                    logger.info(f"Batch {batch_num} successful: {len(converted_records)} records inserted")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                failed_inserts += len(batch)
        
        logger.info(f"Batch insertion complete. Success: {successful_inserts}, Failed: {failed_inserts}")
        
        return {
            'successful': successful_inserts,
            'failed': failed_inserts,
            'total': total_records
        }
    
    def insert_from_json_file(self, json_file_path: str, batch_size: int = 100) -> Dict[str, int]:
        """
        Insert records from a JSON file.
        
        Args:
            json_file_path: Path to JSON file
            batch_size: Number of records to insert per batch
            
        Returns:
            Dictionary with success/failure counts
        """
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            records = data.get('records', [])
            if not records:
                logger.warning(f"No records found in {json_file_path}")
                return {'successful': 0, 'failed': 0, 'total': 0}
            
            logger.info(f"Found {len(records)} records in {json_file_path}")
            return self.insert_many_records(records, batch_size)
            
        except Exception as e:
            logger.error(f"Error reading JSON file {json_file_path}: {e}")
            return {'successful': 0, 'failed': 0, 'total': 0}
    
    def insert_from_parsed_data_directory(self, parsed_data_dir: str = "parsed_data", batch_size: int = 100) -> Dict[str, int]:
        """
        Insert all records from the parsed_data directory.
        
        Args:
            parsed_data_dir: Directory containing parsed JSON files
            batch_size: Number of records to insert per batch
            
        Returns:
            Dictionary with overall success/failure counts
        """
        if not os.path.exists(parsed_data_dir):
            logger.error(f"Parsed data directory not found: {parsed_data_dir}")
            return {'successful': 0, 'failed': 0, 'total': 0}
        
        json_files = [f for f in os.listdir(parsed_data_dir) if f.endswith('.json')]
        
        if not json_files:
            logger.warning(f"No JSON files found in {parsed_data_dir}")
            return {'successful': 0, 'failed': 0, 'total': 0}
        
        total_successful = 0
        total_failed = 0
        total_records = 0
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            json_path = os.path.join(parsed_data_dir, json_file)
            logger.info(f"Processing {json_file}")
            
            result = self.insert_from_json_file(json_path, batch_size)
            
            total_successful += result['successful']
            total_failed += result['failed']
            total_records += result['total']
        
        logger.info(f"All files processed. Total - Success: {total_successful}, Failed: {total_failed}, Total Records: {total_records}")
        
        return {
            'successful': total_successful,
            'failed': total_failed,
            'total': total_records
        }


def main():
    """Main function to demonstrate database insertion."""
    
    # Get credentials from environment variables
    application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    
    if not application_token or not api_endpoint:
        logger.error("Please set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT environment variables")
        return
    
    # Initialize inserter
    inserter = WellWiseDBInserter(application_token, api_endpoint)
    
    # Insert all data from parsed_data directory
    result = inserter.insert_from_parsed_data_directory()
    
    print(f"\n=== DATABASE INSERTION RESULTS ===")
    print(f"Total Records: {result['total']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")
    print(f"Success Rate: {(result['successful']/result['total']*100):.1f}%" if result['total'] > 0 else "No records processed")


if __name__ == "__main__":
    main() 