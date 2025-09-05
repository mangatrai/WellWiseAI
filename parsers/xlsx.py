#!/usr/bin/env python3
"""
XLSX Parser for Facies Interpretation Files

This module parses XLSX files containing geological facies interpretations
from the Volve dataset. These files contain detailed geological classifications
that complement the well log data from LAS/DLIS files.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XLSXParser:
    """Parser for XLSX facies interpretation files."""
    
    def __init__(self):
        """Initialize the XLSX parser."""
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.chat_completion_model = os.getenv('CHAT_COMPLETION_MODEL', 'gpt-4o-mini')
        
        if self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
            self.llm_available = True
            logger.info(f"LLM enhancement available using {self.chat_completion_model}")
        else:
            self.client = None
            self.llm_available = False
            logger.warning("LLM enhancement not available - OPENAI_API_KEY not found")
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse an XLSX facies interpretation file.
        
        Args:
            file_path: Path to the XLSX file
            
        Returns:
            List of dictionaries containing parsed data
        """
        try:
            logger.info(f"Parsing XLSX file: {file_path}")
            
            # Read the Excel file
            df = pd.read_excel(file_path)
            
            if df.empty:
                logger.warning(f"Empty XLSX file: {file_path}")
                return []
            
            # Validate required columns
            required_columns = [
                '* Well UWI', 'Common Well Name', '* Litho Crv Type', 
                '* Source', '* Top Depth (meters)', '* Base Depth (meters)', 
                'Litho Class', 'Rock Percent (%)', 'Original Data Source'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in {file_path}: {missing_columns}")
                return []
            
            # Extract well information
            well_uwi = df['* Well UWI'].iloc[0]
            well_name = df['Common Well Name'].iloc[0]
            
            logger.info(f"Processing well: {well_name} ({well_uwi})")
            logger.info(f"Total records: {len(df)}")
            
            # Parse each row
            parsed_records = []
            for index, row in df.iterrows():
                try:
                    record = self._parse_row(row, well_name, well_uwi, file_path)
                    if record:
                        parsed_records.append(record)
                except Exception as e:
                    logger.error(f"Error parsing row {index}: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(parsed_records)} records")
            return parsed_records
            
        except Exception as e:
            logger.error(f"Error parsing XLSX file {file_path}: {e}")
            return []
    
    def _parse_row(self, row: pd.Series, well_name: str, well_uwi: str, file_path: str = "") -> Optional[Dict[str, Any]]:
        """
        Parse a single row from the XLSX file.
        
        Args:
            row: Pandas Series representing a row
            well_name: Common well name
            well_uwi: Well UWI identifier
            
        Returns:
            Dictionary containing parsed data or None if parsing fails
        """
        try:
            # Basic extraction (direct mappings)
            record = {
                # Well identification
                'well_id': well_name,
                
                # Record type and metadata
                'record_type': 'facies_interpretation',
                'curve_name': str(row['Litho Class']),
                'tool_type': str(row['* Source']),
                'processing_software': str(row['* Source']),
                
                # Depth information
                'depth_start': float(row['* Top Depth (meters)']),
                'depth_end': float(row['* Base Depth (meters)']),
                
                # Geological classification
                'facies_code': str(row['Litho Class']),
                'horizon_name': str(row['Litho Class']),
                
                # Calculated fields
                'sample_interval': float(row['* Base Depth (meters)']) - float(row['* Top Depth (meters)']),
                'num_samples': 1,  # Each row represents one interval
                
                # Metadata - map to canonical fields where possible
                'file_origin': f"xlsx_facies_interpretation:{os.path.basename(file_path)}",
                'processing_date': datetime.now().isoformat(),
                'version': '1.0',
                
                # Quality control
                'qc_flag': True if pd.notna(row['Litho Class']) else False,
                
                # Consolidate non-canonical fields into remarks
                'remarks': f"Source: {row['* Source']}, Type: {row['* Litho Crv Type']}, " + \
                          f"Well UWI: {well_uwi}, Depth Unit: meters, " + \
                          f"Lithology Type: {row['* Litho Crv Type']}, " + \
                          f"Rock Percentage: {row['Rock Percent (%)'] if pd.notna(row['Rock Percent (%)']) else 'N/A'}"
            }
            
            # Add original data source if available
            if pd.notna(row['Original Data Source']):
                record['remarks'] += f", Original Data Source: {row['Original Data Source']}"
            
            # Apply LLM enhancements if available
            if self.llm_available:
                enhanced_record = self._apply_llm_enhancements(record)
                if enhanced_record:
                    record.update(enhanced_record)
            
            return record
            
        except Exception as e:
            logger.error(f"Error parsing row: {e}")
            return None
    
    def _apply_llm_enhancements(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply LLM enhancements to extract additional geological context.
        
        Args:
            record: Parsed record dictionary
            
        Returns:
            Dictionary with LLM-enhanced fields or None if enhancement fails
        """
        try:
            # Prepare context for LLM
            facies_code = record.get('facies_code', '')
            depth_start = record.get('depth_start', 0)
            depth_end = record.get('depth_end', 0)
            well_id = record.get('well_id', '')
            
            prompt = f"""
            Analyze this geological facies interpretation and extract additional context:
            
            Well ID: {well_id}
            Depth Interval: {depth_start}-{depth_end} meters
            Facies Code: {facies_code}
            
            Please extract the following information in JSON format:
            1. field_name: The oil field name (likely "Volve" based on well naming)
            2. country: The country (likely "Norway" based on well naming)
            3. formation_temp: Estimated temperature at depth (use 3°C/100m geothermal gradient) - RETURN ONLY THE FINAL NUMBER, NO CALCULATIONS
            4. formation_press: Estimated pressure at depth (use 10 MPa/km pressure gradient) - RETURN ONLY THE FINAL NUMBER, NO CALCULATIONS
            5. geological_summary: Brief geological interpretation (max 200 characters)
            
            IMPORTANT: Return only valid JSON with final calculated values. Do not include any mathematical expressions, calculations, or reasoning in the JSON. Only return the final numeric results.
            """
            
            response = self.client.chat.completions.create(
                model=self.chat_completion_model,
                messages=[
                    {"role": "system", "content": "You are a geological expert. Extract geological context from facies interpretations and return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            
            # Parse LLM response
            llm_content = response.choices[0].message.content.strip()
            #logger.debug(f"LLM response: {llm_content}")
            
            # Try to extract JSON from response with better error handling
            try:
                enhanced_data = json.loads(llm_content)
                logger.debug(f"LLM enhancement successful for {record.get('well_id')} at {record.get('depth_start')}m")
                
                # Add geological context to remarks
                if 'geological_summary' in enhanced_data:
                    record['remarks'] += f", Geological Context: {enhanced_data['geological_summary']}"
                
                # Return only canonical fields
                canonical_enhanced_data = {}
                for field in ['field_name', 'country', 'formation_temp', 'formation_press']:
                    if field in enhanced_data:
                        canonical_enhanced_data[field] = enhanced_data[field]
                
                return canonical_enhanced_data
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for LLM response: {e}")
                logger.debug(f"Raw LLM response: {llm_content}")
                return None
                
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return None
    
    def validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate a parsed record against canonical schema requirements.
        
        Args:
            record: Parsed record dictionary
            
        Returns:
            True if record is valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = [
                'well_id', 'depth_start', 'depth_end', 'curve_name',
                'record_type', 'data_source'
            ]
            
            for field in required_fields:
                if field not in record or record[field] is None:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate depth values
            if record['depth_start'] >= record['depth_end']:
                logger.warning(f"Invalid depth interval: {record['depth_start']} >= {record['depth_end']}")
                return False
            
            # Validate data types
            if not isinstance(record['depth_start'], (int, float)):
                logger.warning(f"Invalid depth_start type: {type(record['depth_start'])}")
                return False
            
            if not isinstance(record['depth_end'], (int, float)):
                logger.warning(f"Invalid depth_end type: {type(record['depth_end'])}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Record validation error: {e}")
            return False


def parse_xlsx_file(file_path: str, logger=None) -> List[Dict[str, Any]]:
    """
    Convenience function to parse an XLSX file.
    
    Args:
        file_path: Path to the XLSX file
        logger: Optional logger instance
        
    Returns:
        List of dictionaries containing parsed data
    """
    parser = XLSXParser()
    return parser.parse_file(file_path)


if __name__ == "__main__":
    # Test the parser
    test_file = "/Users/mrai/Library/CloudStorage/Box-Box/Volve/PETROPHYSICAL INTERPRETATION/15_9-19 BT2/15_9-19 BT2 Facies.xlsx"
    
    if os.path.exists(test_file):
        print(f"Testing XLSX parser with: {test_file}")
        records = parse_xlsx_file(test_file)
        print(f"Parsed {len(records)} records")
        
        if records:
            print("\nSample record:")
            print(json.dumps(records[0], indent=2))
    else:
        print(f"Test file not found: {test_file}")
