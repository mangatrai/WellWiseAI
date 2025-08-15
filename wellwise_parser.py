#!/usr/bin/env python3
import os
import json
import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

# Load env vars from .env
load_dotenv()

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup structured logging with proper formatting."""
    logger = logging.getLogger('wellwise_parser')
    
    # Convert string log level to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    log_level_constant = level_map.get(log_level.upper(), logging.INFO)
    logger.setLevel(log_level_constant)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_constant)
    
    # File handler
    file_handler = logging.FileHandler('wellwise_parser.log')
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def get_environment_config() -> Dict[str, str]:
    """Get configuration from environment variables with fallbacks."""
    config = {
        'data_directory': os.getenv("DATA_DIRECTORY", "./data"),
        'parsed_directory': os.getenv("PARSED_DIRECTORY", "parsed_data"),
        'unstructured_directory': os.getenv("UNST_PARSED_DATA_DIR", "unstructured_data"),
        'unstructured_file_types': os.getenv("UNSTRUCTURED_FILE_TYPES", "").split(",") if os.getenv("UNSTRUCTURED_FILE_TYPES") else [],
        'structured_file_types': os.getenv("STRUCTURED_FILE_TYPES", ".las,.dlis,.csv").split(","),
        'max_workers': int(os.getenv("MAX_WORKERS", "4")),
        'retry_attempts': int(os.getenv("RETRY_ATTEMPTS", "3")),
        'timeout': int(os.getenv("TIMEOUT_SECONDS", "300")),
        'log_level': os.getenv("LOG_LEVEL", "INFO")
    }
    return config

def is_unstructured_file(file_path: str, config: Dict[str, str]) -> bool:
    """Check if a file should be processed as unstructured based on config."""
    if not config.get('unstructured_file_types'):
        return False
    
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in config['unstructured_file_types']

def get_parser_for_extension(ext: str):
    """Get the appropriate parser function for a file extension."""
    parser_map = {
        ".dlis": parse_dlis,
        ".las": parse_las, 
        ".csv": parse_csv_file,
    }
    return parser_map.get(ext)

def validate_json_output(data: List[Dict]) -> bool:
    """Validate JSON output structure and data integrity."""
    if not isinstance(data, list):
        return False
    
    if not data:
        return True  # Empty list is valid
    
    # Check that all items are dictionaries
    if not all(isinstance(item, dict) for item in data):
        return False
    
    # Check for required fields in first record (assuming all records have same structure)
    required_fields = ['well_id', 'file_origin', 'record_type', 'curve_name']
    if not all(field in data[0] for field in required_fields):
        return False
    
    return True

def create_output_filename(well_id: str, curve_name: str, original_name: str) -> str:
    """Create meaningful output filename."""
    # Clean the well_id and curve_name for filename safety
    safe_well_id = "".join(c for c in well_id if c.isalnum() or c in ('-', '_')).rstrip()
    safe_curve_name = "".join(c for c in curve_name if c.isalnum() or c in ('-', '_')).rstrip()
    
    # Use original name as fallback if well_id or curve_name are empty
    base_name = original_name.rsplit('.', 1)[0]
    
    if safe_well_id and safe_curve_name:
        return f"{safe_well_id}_{safe_curve_name}.json"
    elif safe_well_id:
        return f"{safe_well_id}_{base_name}.json"
    else:
        return f"{base_name}.json"

def process_single_file(file_path: str, parser_func, logger: logging.Logger, 
                       config: Dict[str, str]) -> Tuple[bool, str, Dict]:
    """
    Process a single file with error handling and validation.
    
    Returns:
        Tuple of (success: bool, message: str, data: Dict)
    """
    try:
        # Special handling for CSV files
        if file_path.lower().endswith('.csv'):
            # CSV parser generates multiple batch files
            output_files = parser_func(file_path, config['parsed_directory'])
            
            # Create metadata for CSV processing
            output_data = {
                'metadata': {
                    'processing_timestamp': datetime.utcnow().isoformat(),
                    'source_file': file_path,
                    'output_files': output_files,
                    'parser_version': '1.0.0',
                    'file_type': 'csv'
                },
                'records': []  # CSV records are in separate batch files
            }
            
            # Write metadata file
            output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_metadata.json"
            output_path = os.path.join(config['parsed_directory'], output_filename)
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            return True, f"Successfully processed CSV {file_path} → {len(output_files)} batch files", output_data
        
        # Regular handling for other file types
        # Parse the file - pass logger to parser function
        records = parser_func(file_path, logger)
        
        # Validate the output
        if not validate_json_output(records):
            return False, f"Invalid JSON structure for {file_path}", {}
        
        # Create output filename
        if records and len(records) > 0:
            well_id = records[0].get('well_id', '')
            curve_name = records[0].get('curve_name', '')
            original_name = os.path.basename(file_path)
            output_filename = create_output_filename(well_id, curve_name, original_name)
        else:
            output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}.json"
        
        # Write to JSON file
        output_path = os.path.join(config['parsed_directory'], output_filename)
        
        # Add metadata to the output
        output_data = {
            'metadata': {
                'processing_timestamp': datetime.utcnow().isoformat(),
                'source_file': file_path,
                'record_count': len(records),
                'parser_version': '1.0.0'
            },
            'records': records
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        return True, f"Successfully processed {file_path} → {output_path}", output_data
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        return False, f"Failed to process {file_path}: {str(e)}", {}

def process_unstructured_file(file_path: str, logger: logging.Logger, 
                            config: Dict[str, str]) -> Tuple[bool, str, Dict]:
    """
    Process a single unstructured file using UnstructuredParser.
    
    Returns:
        Tuple of (success: bool, message: str, data: Dict)
    """
    try:
        # Create UnstructuredParser instance with configuration
        parser = UnstructuredParser(file_path, config['unstructured_file_types'])
        
        # Check if parser can handle this file
        if not parser.can_parse():
            return False, f"Cannot parse unstructured file: {file_path}", {}
        
        # Parse the file
        parsed_data = parser.parse()
        
        if parsed_data.error:
            return False, f"Error parsing unstructured file {file_path}: {parsed_data.error}", {}
        
        # Generate contextual documents
        contextual_docs = parser.generate_contextual_documents()
        
        # Save to unstructured directory
        parent_dir = os.path.basename(os.path.dirname(file_path))
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{parent_dir}_{base_name}_unstructured.json"
        output_path = os.path.join(config['unstructured_directory'], output_filename)
        
        # Prepare output data
        output_data = {
            'metadata': {
                'processing_timestamp': datetime.utcnow().isoformat(),
                'source_file': os.path.basename(file_path),
                'contextual_documents_count': len(contextual_docs)
            },
            'contextual_documents': [
                {
                    'content': doc.content,
                    'well_id': doc.metadata.get('well_id', ''),
                    'depth_references': doc.metadata.get('depth_references', []),
                    'technical_terms': doc.metadata.get('technical_terms', []),
                    'mathematical_formulas': doc.metadata.get('mathematical_formulas', []),
                    'geological_context': doc.metadata.get('geological_context', {}),
                    'curve_names': doc.metadata.get('curve_names', []),
                    'document_type': doc.metadata.get('document_type', doc.document_type),
                    'source': os.path.basename(doc.source),
                    'timestamp': doc.timestamp,
                    'metadata': {
                        'chunk_id': doc.metadata.get('chunk_id', 0),
                        'total_chunks': doc.metadata.get('total_chunks', 0),
                        'element_type': doc.metadata.get('element_type', ''),
                        'chunk_type': doc.metadata.get('chunk_type', '')
                    }
                } for doc in contextual_docs
            ]
        }
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        return True, f"Successfully processed unstructured {file_path} → {output_path}", output_data
        
    except Exception as e:
        logger.error(f"Error processing unstructured file {file_path}: {str(e)}", exc_info=True)
        return False, f"Failed to process unstructured file {file_path}: {str(e)}", {}

# Import your parsers
from parsers.dlis import parse_dlis
from parsers.las import parse_las
from parsers.csv_parser import parse_csv_file
from parsers.unstructured import UnstructuredParser

def main():
    # Get configuration first
    config = get_environment_config()
    
    # Setup logging with environment log level
    logger = setup_logging(config['log_level'])
    logger.info("Starting WellWise parser")
    logger.info(f"Configuration: {config}")
    
    # Validate data directory
    data_directory = config['data_directory']
    if not os.path.exists(data_directory):
        logger.error(f"Data directory does not exist: {data_directory}")
        raise ValueError(f"Data directory does not exist: {data_directory}")
    
    # Create output directories
    os.makedirs(config['parsed_directory'], exist_ok=True)
    os.makedirs(config['unstructured_directory'], exist_ok=True)
    logger.info(f"Structured output directory: {config['parsed_directory']}")
    logger.info(f"Unstructured output directory: {config['unstructured_directory']}")
    
    # Collect all files to process - separate structured and unstructured
    structured_files = []
    unstructured_files = []
    unmatched_files = []
    
    for root, _, files in os.walk(data_directory):
        for fname in files:
            file_path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            
            # Check if it's an unstructured file first
            if is_unstructured_file(file_path, config):
                unstructured_files.append(file_path)
                logger.debug(f"Unstructured file: {file_path}")
            # Then check if it's a structured file using config
            elif ext in config['structured_file_types']:
                parser_func = get_parser_for_extension(ext)
                if parser_func:
                    structured_files.append((file_path, parser_func))
                    logger.debug(f"Structured file: {file_path}")
                else:
                    unmatched_files.append(file_path)
                    logger.warning(f"Structured file type configured but no parser available: {file_path} (extension: {ext})")
            # Handle unmatched files
            else:
                unmatched_files.append(file_path)
                logger.warning(f"Unmatched file type (no parser available): {file_path} (extension: {ext})")
    
    logger.info(f"Found {len(structured_files)} structured files to process")
    logger.info(f"Found {len(unstructured_files)} unstructured files to process")
    logger.info(f"Found {len(unmatched_files)} unmatched files (ignored)")
    
    if unmatched_files:
        logger.warning(f"Unmatched files that will be ignored: {unmatched_files}")
    
    # Process structured files (existing logic)
    if structured_files:
        logger.info("Processing structured files...")
        successful_files = 0
        failed_files = 0
        
        with tqdm(total=len(structured_files), desc="Processing structured files") as pbar:
            # Use ThreadPoolExecutor for parallel processing
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=config['max_workers']
            ) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(process_single_file, file_path, parser_func, logger, config): file_path
                    for file_path, parser_func in structured_files
                }
                
                # Process completed tasks
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        success, message, data = future.result()
                        if success:
                            successful_files += 1
                            logger.info(message)
                        else:
                            failed_files += 1
                            logger.error(message)
                    except Exception as e:
                        failed_files += 1
                        logger.error(f"Unexpected error processing {file_path}: {str(e)}", exc_info=True)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': successful_files,
                        'Failed': failed_files
                    })
        
        logger.info(f"Structured files processing complete. Success: {successful_files}, Failed: {failed_files}")
    
    # Process unstructured files
    if unstructured_files:
        logger.info("Processing unstructured files...")
        successful_files = 0
        failed_files = 0
        
        with tqdm(total=len(unstructured_files), desc="Processing unstructured files") as pbar:
            # Use ThreadPoolExecutor for parallel processing
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=config['max_workers']
            ) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(process_unstructured_file, file_path, logger, config): file_path
                    for file_path in unstructured_files
                }
                
                # Process completed tasks
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        success, message, data = future.result()
                        if success:
                            successful_files += 1
                            logger.info(message)
                        else:
                            failed_files += 1
                            logger.error(message)
                    except Exception as e:
                        failed_files += 1
                        logger.error(f"Unexpected error processing unstructured file {file_path}: {str(e)}", exc_info=True)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': successful_files,
                        'Failed': failed_files
                    })
        
        logger.info(f"Unstructured files processing complete. Success: {successful_files}, Failed: {failed_files}")
    
    if not structured_files and not unstructured_files:
        logger.warning("No files found to process")
        return

if __name__ == "__main__":
    main()
