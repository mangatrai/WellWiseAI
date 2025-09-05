#!/usr/bin/env python3
"""
Well Survey Parser for No-Extension Files

This module parses well survey files that have no file extension.
These files contain directional survey data including MD, TVD, inclination, azimuth,
and coordinates for well trajectory analysis.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurveyParser:
    """Parser for well survey files (no extension)."""
    
    def __init__(self):
        """Initialize the survey parser."""
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.chat_completion_model = os.getenv('CHAT_COMPLETION_MODEL', 'gpt-4o-mini')
        
        if self.openai_api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.openai_api_key)
            self.llm_available = True
            logger.info(f"LLM enhancement available using {self.chat_completion_model}")
        else:
            self.client = None
            self.llm_available = False
            logger.warning("LLM enhancement not available - OPENAI_API_KEY not found")
    
    def is_survey_file(self, file_path: str) -> bool:
        """
        Check if a file is a well survey file based on filename and content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if it's a survey file, False otherwise
        """
        try:
            # Check filename patterns
            filename = os.path.basename(file_path).lower()
            survey_patterns = [
                'plan', 'actual', 'survey', 'trajectory', 'directional',
                'wellbore', 'drilling', 'md', 'tvd', 'inclination', 'azimuth'
            ]
            
            if any(pattern in filename for pattern in survey_patterns):
                return True
            
            # Check content patterns
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = f.read(2000)  # Read first 2KB
                
                # Look for survey-specific content patterns
                survey_content_patterns = [
                    r'HEADER INFORMATION',
                    r'WELL NAME:',
                    r'MD\s+Inc\s+Azim\s+TVD',
                    r'Measured Depth',
                    r'True Vertical Depth',
                    r'Inclination',
                    r'Azimuth',
                    r'Survey Name:',
                    r'Calculation Method:',
                    r'Surface Latitude:',
                    r'Surface Longitude:'
                ]
                
                for pattern in survey_content_patterns:
                    if re.search(pattern, first_lines, re.IGNORECASE):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if file is survey file {file_path}: {e}")
            return False
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a well survey file.
        
        Args:
            file_path: Path to the survey file
            
        Returns:
            List of dictionaries containing parsed survey data
        """
        try:
            logger.info(f"Parsing survey file: {file_path}")
            
            # Read the entire file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Empty survey file: {file_path}")
                return []
            
            # Extract header information
            header_info = self._extract_header_info(content)
            
            # Extract survey data
            survey_data = self._extract_survey_data(content)
            
            if not survey_data:
                logger.warning(f"No survey data found in {file_path}")
                return []
            
            logger.info(f"Processing well: {header_info.get('well_name', 'Unknown')}")
            logger.info(f"Total survey points: {len(survey_data)}")
            
            # Parse each survey point
            parsed_records = []
            for index, survey_point in enumerate(survey_data):
                try:
                    record = self._parse_survey_point(survey_point, header_info, file_path, index)
                    if record:
                        parsed_records.append(record)
                except Exception as e:
                    logger.error(f"Error parsing survey point {index}: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(parsed_records)} survey points")
            return parsed_records
            
        except Exception as e:
            logger.error(f"Error parsing survey file {file_path}: {e}")
            return []
    
    def _extract_header_info(self, content: str) -> Dict[str, Any]:
        """Extract header information from survey file content."""
        header_info = {}
        
        # Extract well information
        well_name_match = re.search(r'WELL NAME:\s*([^\n]+)', content, re.IGNORECASE)
        if well_name_match:
            header_info['well_name'] = well_name_match.group(1).strip()
        
        field_match = re.search(r'FIELD:\s*([^\n]+)', content, re.IGNORECASE)
        if field_match:
            header_info['field_name'] = field_match.group(1).strip()
        
        company_match = re.search(r'COMPANY:\s*([^\n]+)', content, re.IGNORECASE)
        if company_match:
            header_info['company'] = company_match.group(1).strip()
        
        # Extract coordinates
        lat_match = re.search(r'Surface Latitude:\s*([^\n]+)', content, re.IGNORECASE)
        if lat_match:
            header_info['surface_latitude'] = lat_match.group(1).strip()
        
        lon_match = re.search(r'Surface Longitude:\s*([^\n]+)', content, re.IGNORECASE)
        if lon_match:
            header_info['surface_longitude'] = lon_match.group(1).strip()
        
        # Extract dates
        survey_date_match = re.search(r'Survey Date:\s*([^\n]+)', content, re.IGNORECASE)
        if survey_date_match:
            header_info['survey_date'] = survey_date_match.group(1).strip()
        
        extraction_date_match = re.search(r'Extraction Date:\s*([^\n]+)', content, re.IGNORECASE)
        if extraction_date_match:
            header_info['extraction_date'] = extraction_date_match.group(1).strip()
        
        return header_info
    
    def _extract_survey_data(self, content: str) -> List[Dict[str, Any]]:
        """Extract survey data points from file content."""
        survey_data = []
        
        # Look for survey data table
        # Pattern: MD, Inc, Azim, TVD, X-Offset, Y-Offset, UTM E/W, UTM N/S, DLS
        survey_pattern = r'(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)'
        
        # Find the survey data section
        lines = content.split('\n')
        in_survey_section = False
        
        for line in lines:
            # Check if we're entering survey data section
            if re.search(r'SURVEY LIST|MD\s+Inc\s+Azim\s+TVD', line, re.IGNORECASE):
                in_survey_section = True
                continue
            
            if in_survey_section:
                # Skip header lines and empty lines
                if re.search(r'MD\s+Inc\s+Azim|m RKB\s+deg\s+deg|^-+$', line, re.IGNORECASE):
                    continue
                
                if not line.strip():
                    continue
                
                # Try to match survey data pattern
                match = re.match(survey_pattern, line.strip())
                if match:
                    survey_point = {
                        'md': float(match.group(1)),
                        'inclination': float(match.group(2)),
                        'azimuth': float(match.group(3)),
                        'tvd': float(match.group(4)),
                        'x_offset': float(match.group(5)),
                        'y_offset': float(match.group(6)),
                        'utm_east': float(match.group(7)),
                        'utm_north': float(match.group(8)),
                        'dls': float(match.group(9))
                    }
                    survey_data.append(survey_point)
        
        return survey_data
    
    def _parse_survey_point(self, survey_point: Dict[str, Any], header_info: Dict[str, Any], 
                           file_path: str, index: int) -> Optional[Dict[str, Any]]:
        """Parse a single survey point into canonical format."""
        try:
            # Basic extraction (direct mappings)
            record = {
                # Well identification
                'well_id': header_info.get('well_name', 'Unknown'),
                
                # Record type and metadata
                'record_type': 'well_survey',
                'curve_name': 'directional_survey',
                'tool_type': 'directional_survey',
                'processing_software': 'survey_parser',
                
                # Depth information
                'depth_start': survey_point['md'],
                'plan_md': survey_point['md'],
                'plan_tvd': survey_point['tvd'],
                
                # Directional data
                'plan_inclination': survey_point['inclination'],
                'plan_azimuth': survey_point['azimuth'],
                
                # Calculated fields
                'sample_interval': 0.0,  # Each point is discrete
                'num_samples': 1,
                
                # Metadata
                'file_origin': f"survey_parser:{os.path.basename(file_path)}",
                'processing_date': datetime.now(),
                'version': '1.0',
                
                # Quality control
                'qc_flag': True if survey_point['dls'] < 3.0 else False,  # DLS < 3°/30m is good
                'remarks': f"Survey Point {index + 1}, DLS: {survey_point['dls']:.2f}°/30m, " + \
                          f"X-Offset: {survey_point['x_offset']:.2f}m, Y-Offset: {survey_point['y_offset']:.2f}m, " + \
                          f"UTM E: {survey_point['utm_east']:.2f}m, UTM N: {survey_point['utm_north']:.2f}m"
            }
            
            # Add field name if available
            if 'field_name' in header_info:
                record['field_name'] = header_info['field_name']
            
            # Add company/country if available
            if 'company' in header_info:
                company = header_info['company']
                if 'norway' in company.lower() or 'statoil' in company.lower():
                    record['country'] = 'Norway'
            
            # Apply LLM enhancements if available
            if self.llm_available:
                enhanced_record = self._apply_llm_enhancements(record, header_info)
                if enhanced_record:
                    record.update(enhanced_record)
            
            return record
            
        except Exception as e:
            logger.error(f"Error parsing survey point: {e}")
            return None
    
    def _apply_llm_enhancements(self, record: Dict[str, Any], header_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply LLM enhancements to extract additional context."""
        try:
            # Prepare context for LLM
            well_id = record.get('well_id', '')
            md = record.get('plan_md', 0)
            tvd = record.get('plan_tvd', 0)
            inclination = record.get('plan_inclination', 0)
            azimuth = record.get('plan_azimuth', 0)
            field_name = header_info.get('field_name', '')
            
            prompt = f"""
            Analyze this well survey data and extract additional context:
            
            Well ID: {well_id}
            Field: {field_name}
            Measured Depth: {md}m
            True Vertical Depth: {tvd}m
            Inclination: {inclination}°
            Azimuth: {azimuth}°
            
            Please extract the following information in JSON format:
            1. field_name: The oil field name (likely "{field_name}" if provided)
            2. country: The country (likely "Norway" based on company info)
            3. formation_temp: Estimated temperature at depth (use 3°C/100m geothermal gradient) - RETURN ONLY THE FINAL NUMBER, NO CALCULATIONS
            4. formation_press: Estimated pressure at depth (use 10 MPa/km pressure gradient) - RETURN ONLY THE FINAL NUMBER, NO CALCULATIONS
            5. survey_summary: Brief survey interpretation (max 200 characters)
            
            IMPORTANT: Return only valid JSON with final calculated values. Do not include any mathematical expressions, calculations, or reasoning in the JSON. Only return the final numeric results.
            """
            
            response = self.client.chat.completions.create(
                model=self.chat_completion_model,
                messages=[
                    {"role": "system", "content": "You are a well survey expert. Extract survey context and return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            # Parse LLM response
            llm_content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from response with better error handling
            try:
                enhanced_data = json.loads(llm_content)
                logger.debug(f"LLM enhancement successful for {record.get('well_id')} at {record.get('plan_md')}m")
                
                # Add survey context to remarks
                if 'survey_summary' in enhanced_data:
                    record['remarks'] += f", Survey Context: {enhanced_data['survey_summary']}"
                
                # Return only canonical fields
                canonical_enhanced_data = {}
                for field in ['field_name', 'country', 'formation_temp', 'formation_press']:
                    if field in enhanced_data:
                        canonical_enhanced_data[field] = enhanced_data[field]
                
                return canonical_enhanced_data
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed for LLM response: {e}")
                logger.debug(f"Raw LLM response: {llm_content}")
                return None
                
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return None


def parse_survey_file(file_path: str, logger=None) -> List[Dict[str, Any]]:
    """
    Convenience function to parse a survey file.
    
    Args:
        file_path: Path to the survey file
        logger: Optional logger instance
        
    Returns:
        List of dictionaries containing parsed survey data
    """
    parser = SurveyParser()
    return parser.parse_file(file_path)


def is_survey_file(file_path: str) -> bool:
    """
    Check if a file is a well survey file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if it's a survey file, False otherwise
    """
    parser = SurveyParser()
    return parser.is_survey_file(file_path)


if __name__ == "__main__":
    # Test the parser
    test_file = "/Users/mrai/Library/CloudStorage/Box-Box/Volve/Volve F_F-12_F-12_F-12_ACTUAL"
    
    if os.path.exists(test_file):
        print(f"Testing survey parser with: {test_file}")
        
        # First check if it's a survey file
        if is_survey_file(test_file):
            print("✅ File identified as survey file")
            records = parse_survey_file(test_file)
            print(f"Parsed {len(records)} survey points")
            
            if records:
                print("\nSample record:")
                print(json.dumps(records[0], indent=2))
        else:
            print("❌ File not identified as survey file")
    else:
        print(f"Test file not found: {test_file}")
