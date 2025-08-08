#!/usr/bin/env python3
"""
CSV Parser for WellWiseAI
Parses drilling data CSV files and converts to canonical JSON format.
"""

import pandas as pd
import json
import os
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
from schema.fields import CANONICAL_FIELDS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSV to canonical field mappings (39 fields) - using list to handle duplicate CSV column names
# Each tuple is (csv_column_name, canonical_field_name, priority)
# Priority determines which mapping takes precedence when CSV column appears multiple times
# Lower priority number = higher precedence
CSV_TO_CANONICAL_MAPPINGS = [
    # Primary mappings (priority 1) - highest precedence
    ('Measured Depth m', 'depth_start', 1),
    ('Hole Depth (TVD) m', 'plan_tvd', 1), 
    ('Rate of Penetration m/h', 'rop', 1),
    ('Weight on Bit kkgf', 'weight_on_bit', 1),
    ('Average Surface Torque kN.m', 'torque', 1),
    ('Total Downhole RPM rpm', 'sample_mean', 1),
    ('ARC Gamma Ray (BH corrected) gAPI', 'sp_curves', 1),
    ('ARC Attenuation Resistivity 28 inch at 2 MHz ohm.m', 'resistivity_shallow', 1),
    ('Mud Density In g/cm3', 'mud_weight_actual', 1),
    ('Average Standpipe Pressure kPa', 'pump_pressure', 1),
    ('Annular Temperature degC', 'formation_temp', 1),
    ('ARC Annular Pressure kPa', 'formation_press', 1),
    ('Utrasonic Caliper, Average Diameter, Computed DH cm', 'caliper', 1),
    ('Bulk Density, Bottom, Computed DH g/cm3', 'vshale', 1),
    ('Gas (avg) %', 'gas_oil_ratio', 1),
    ('MWD Shock Risk unitless', 'qc_flag', 1),
    ('Total SPM 1/min', 'sample_interval', 1),
    ('Pump Time h', 'num_samples', 1),
    ('Pass Name unitless', 'tool_type', 1),
    ('MWD Vibration X-Axis ft/s2', 'seismic_sample_rate', 1),
    ('MWD Vibration Lateral ft/s2', 'seismic_trace_count', 1),
    ('MWD Vibration Torsional kN.m', 'vp', 1),
    
    # Secondary mappings (priority 2) - for duplicate CSV columns
    ('Bit run number unitless', 'sample_stddev', 1),  # Primary
    ('Bit run number unitless', 'file_origin', 2),    # Secondary
    ('Mud Flow In L/min', 'mud_flow_rate', 1),       # Primary
    ('Mud Flow In L/min', 'production_rate', 2),     # Secondary
    ('Corr. Drilling Exponent unitless', 'permeability', 1),      # Primary
    ('Corr. Drilling Exponent unitless', 'core_permeability', 2), # Secondary
    ('Density Porosity from ROBB_RT m3/m3', 'porosity', 1),      # Primary
    ('Density Porosity from ROBB_RT m3/m3', 'water_saturation', 2), # Secondary
    ('ARC Phase Shift Resistivity 40 inch at 2 MHz ohm.m', 'resistivity_deep', 1),    # Primary
    ('ARC Phase Shift Resistivity 40 inch at 2 MHz ohm.m', 'water_resistivity', 2),   # Secondary
    ('ARC Phase Shift Resistivity 40 inch at 2 MHz ohm.m', 'archie_a', 3),            # Tertiary
    ('ARC Phase Shift Resistivity 28 inch at 2 MHz ohm.m', 'resistivity_medium', 1),  # Primary
    ('ARC Phase Shift Resistivity 28 inch at 2 MHz ohm.m', 'archie_m', 2),            # Secondary
    ('nameWellbore', 'field_name', 1),    # Primary
    ('nameWellbore', 'country', 2),       # Secondary
    ('name', 'service_company', 1),       # Primary
    ('name', 'state_province', 2),        # Secondary
]

def parse_csv_file(csv_file_path: str, output_dir: str, records_per_file: int = 1000) -> List[str]:
    """
    Parse CSV file and convert to canonical JSON format.
    
    Args:
        csv_file_path: Path to the CSV file
        output_dir: Directory to save JSON files
        records_per_file: Number of records per JSON file
    
    Returns:
        List of generated JSON file paths
    """
    logger.info(f"Parsing CSV file: {csv_file_path}")
    
    # Validate that all canonical fields we're mapping to exist in the schema
    mapped_canonical_fields = [mapping[1] for mapping in CSV_TO_CANONICAL_MAPPINGS]
    invalid_fields = [field for field in mapped_canonical_fields if field not in CANONICAL_FIELDS]
    if invalid_fields:
        raise ValueError(f"Invalid canonical fields in mappings: {invalid_fields}")
    
    # Read CSV file, skip the first column (serial number)
    df = pd.read_csv(csv_file_path)
    df = df.drop(columns=['Unnamed: 0'])  # Remove serial number column
    
    logger.info(f"CSV has {len(df)} records and {len(df.columns)} data columns")
    
    # Get all column names for remarks mapping
    all_columns = df.columns.tolist()
    
    # Create a mapping of CSV columns to their highest priority canonical field
    # FIXED: Lower priority number = higher precedence
    csv_to_canonical_map = {}
    for csv_col, canonical_field, priority in CSV_TO_CANONICAL_MAPPINGS:
        if csv_col not in csv_to_canonical_map or priority < csv_to_canonical_map[csv_col][1]:
            csv_to_canonical_map[csv_col] = (canonical_field, priority)
    
    canonical_columns = list(csv_to_canonical_map.keys())
    remarks_columns = [col for col in all_columns if col not in canonical_columns]
    
    logger.info(f"Canonical mappings: {len(canonical_columns)}")
    logger.info(f"Remarks fields: {len(remarks_columns)}")
    
    # Extract well ID from first row's nameWellbore column
    sample_row = df.iloc[0]
    name_wellbore = sample_row.get('nameWellbore', None)
    well_id = extract_well_id_from_nameWellbore(name_wellbore, csv_file_path)
    
    # Get current timestamp for processing_date
    processing_date = datetime.now().isoformat() + 'Z'
    
    # Process records in batches
    total_records = len(df)
    num_files = (total_records + records_per_file - 1) // records_per_file
    
    output_files = []
    
    for file_num in range(num_files):
        start_idx = file_num * records_per_file
        end_idx = min((file_num + 1) * records_per_file, total_records)
        
        logger.info(f"Processing records {start_idx+1} to {end_idx} (file {file_num+1}/{num_files})")
        
        batch_records = []
        
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            record = create_canonical_record(row, csv_to_canonical_map, remarks_columns, well_id, processing_date)
            if record:
                batch_records.append(record)
        
        # Save batch to JSON file with metadata structure (compatible with database inserter)
        output_filename = f"csv_parsed_batch_{file_num+1:03d}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create the expected structure with metadata and records
        output_data = {
            'metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'source_file': csv_file_path,
                'record_count': len(batch_records),
                'parser_version': '1.0.0',
                'batch_number': file_num + 1,
                'file_type': 'csv'
            },
            'records': batch_records
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        output_files.append(output_path)
        logger.info(f"Saved {len(batch_records)} records to {output_path}")
    
    logger.info(f"Parsing complete. Generated {len(output_files)} files.")
    return output_files

def extract_well_id_from_nameWellbore(name_wellbore: str, file_path: str = None) -> str:
    """
    Extract well ID from nameWellbore column.
    
    Args:
        name_wellbore: Value from nameWellbore column (e.g., "15/9-F-15A - Main Wellbore")
        file_path: Fallback file path for filename-based extraction
    
    Returns:
        Well ID in original format (e.g., "15/9-F-15A")
    """
    if name_wellbore and isinstance(name_wellbore, str):
        # Split on " - " and take the first part
        parts = name_wellbore.split(" - ")
        if len(parts) > 0 and parts[0].strip():
            return parts[0].strip()
    
    # Simple default if nameWellbore is not available
    return 'well_id'

def create_canonical_record(row: pd.Series, csv_to_canonical_map: Dict[str, Tuple[str, int]], 
                          remarks_columns: List[str], well_id: str, processing_date: str) -> Dict[str, Any]:
    """
    Create a canonical record from a CSV row.
    
    Args:
        row: Pandas Series representing a CSV row
        csv_to_canonical_map: Dictionary mapping CSV columns to (canonical_field, priority)
        remarks_columns: List of columns to add to remarks
        well_id: Well identifier
        processing_date: Processing timestamp
    
    Returns:
        Dictionary representing canonical record
    """
    record = {
        'well_id': well_id,
        'record_type': 'drilling',
        'curve_name': 'drilling_parameters',
        'depth_start': None,
        'acquisition_date': None,
        'processing_date': processing_date
    }
    
    # Map canonical fields using priority system
    for csv_col, (canonical_field, priority) in csv_to_canonical_map.items():
        if csv_col in row and pd.notna(row[csv_col]):
            value = row[csv_col]
            
            # Special handling for nameWellbore field
            if csv_col == 'nameWellbore' and canonical_field == 'field_name':
                # Extract the part after " - " for field_name
                if value and isinstance(value, str) and " - " in value:
                    parts = value.split(" - ")
                    if len(parts) > 1:
                        record[canonical_field] = parts[1].strip()
                    else:
                        record[canonical_field] = value
                else:
                    record[canonical_field] = value
            # Handle data type conversions
            elif canonical_field == 'depth_start':
                try:
                    record['depth_start'] = float(value) if value is not None else None
                except (ValueError, TypeError):
                    record['depth_start'] = None
            elif canonical_field in ['sample_mean', 'sample_stddev', 'rop', 'weight_on_bit', 'torque', 
                                   'plan_tvd', 'mud_flow_rate', 'mud_weight_actual', 'pump_pressure',
                                   'formation_temp', 'formation_press', 'porosity', 'caliper', 'vshale',
                                   'water_resistivity', 'water_saturation', 'gas_oil_ratio', 'production_rate',
                                   'sample_interval', 'permeability', 'archie_a', 'archie_m', 'core_permeability',
                                   'seismic_sample_rate', 'vp', 'resistivity_deep', 'resistivity_medium', 
                                   'resistivity_shallow', 'sp_curves']:
                try:
                    record[canonical_field] = float(value) if value is not None else None
                except (ValueError, TypeError):
                    record[canonical_field] = None
            elif canonical_field == 'qc_flag':
                try:
                    record[canonical_field] = bool(value) if value is not None else False
                except (ValueError, TypeError):
                    record[canonical_field] = False
            elif canonical_field in ['num_samples', 'seismic_trace_count']:
                try:
                    # Convert to int, handling float values by rounding first
                    if value is not None:
                        record[canonical_field] = int(round(float(value)))
                    else:
                        record[canonical_field] = None
                except (ValueError, TypeError):
                    record[canonical_field] = None
            else:
                # String fields
                record[canonical_field] = str(value) if value is not None else None
    
    # Build remarks from remaining fields
    remarks_parts = []
    for col in remarks_columns:
        if col in row and pd.notna(row[col]):
            value = row[col]
            # Clean column name for remarks
            clean_col = col.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
            remarks_parts.append(f"{clean_col}: {value}")
    
    if remarks_parts:
        record['remarks'] = "; ".join(remarks_parts)
    
    return record

def main():
    """Main function for CSV parsing."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python csv_parser.py <csv_file_path> <output_directory>")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file not found: {csv_file_path}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        output_files = parse_csv_file(csv_file_path, output_dir)
        logger.info(f"Successfully parsed CSV. Output files: {output_files}")
    except Exception as e:
        logger.error(f"Error parsing CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 