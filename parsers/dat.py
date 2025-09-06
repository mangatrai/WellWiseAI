#!/usr/bin/env python3
"""
Well Picks Parser for .dat files
Parses well picks data containing formation tops and geological horizons
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from parsers.base_parser import BaseParser
from dotenv import load_dotenv
from utils import LLMClient

class DatParser(BaseParser):
    """Parser for well picks .dat files containing formation tops data"""
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.logger = logging.getLogger(__name__)
        self.supported_extensions = ['.dat']
        self.record_count = 0  # Counter for LLM enhancement limit
        
        # Load environment variables
        load_dotenv()
        
        # Initialize hybrid LLM client
        try:
            self.llm_client = LLMClient()
            self.llm_available = self.llm_client.is_available()
            if self.llm_available:
                backend_info = self.llm_client.get_backend_info()
                self.logger.info(f"LLM enhancement available using {backend_info['backend']}")
            else:
                self.logger.warning("LLM enhancement not available")
        except Exception as e:
            self.llm_client = None
            self.llm_available = False
            self.logger.error(f"Failed to initialize LLM client: {e}")
        
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file"""
        if not super().can_parse(file_path):
            return False
            
        # Check if file contains well picks data
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = [f.readline().strip() for _ in range(10)]
                
            # Look for well picks indicators
            content = ' '.join(first_lines).lower()
            return any(indicator in content for indicator in [
                'well name', 'surface name', 'md', 'tvd', 'tvdss', 'twt', 'dip', 'azi'
            ])
        except Exception as e:
            self.logger.warning(f"Error checking if file can be parsed: {e}")
            return False
    
    def extract_well_id(self, well_name: str) -> str:
        """Extract well ID from well name"""
        # Clean up well name and extract ID
        well_id = well_name.strip()
        
        # Remove any extra whitespace or special characters
        well_id = re.sub(r'\s+', ' ', well_id)
        
        return well_id
    
    def parse_quality_flags(self, qlf: str) -> str:
        """Parse quality flags and return description"""
        if not qlf or qlf.strip() == '':
            return ""
            
        qlf = qlf.strip()
        flag_descriptions = {
            'ER': 'Eroded',
            'FP': 'Faulted pick', 
            'FO': 'Faulted out',
            'NL': 'Not logged',
            'NR': 'Not reached'
        }
        
        return flag_descriptions.get(qlf, f"Unknown flag: {qlf}")
    
    def create_remarks(self, qlf: str, twt: str, tvdss: str, obs_num: str, easting: str, northing: str) -> str:
        """Create remarks field from quality flags and additional data"""
        remarks_parts = []
        
        # Add quality flag description
        if qlf and qlf.strip():
            qlf_desc = self.parse_quality_flags(qlf)
            remarks_parts.append(f"Quality: {qlf_desc}")
        
        # Add TWT if available
        if twt and twt.strip():
            remarks_parts.append(f"TWT: {twt}")
            
        # Add TVDSS if available
        if tvdss and tvdss.strip():
            remarks_parts.append(f"TVDSS: {tvdss}")
            
        # Add observation number if available
        if obs_num and obs_num.strip():
            remarks_parts.append(f"Obs#: {obs_num}")
            
        # Add coordinates if available
        if easting and easting.strip():
            remarks_parts.append(f"Easting: {easting}")
        if northing and northing.strip():
            remarks_parts.append(f"Northing: {northing}")
            
        return "; ".join(remarks_parts) if remarks_parts else ""
    
    def parse_numeric_value(self, value: str) -> Optional[float]:
        """Parse numeric value, handling empty strings and invalid values"""
        if not value or value.strip() == '':
            return None
            
        try:
            return float(value.strip())
        except (ValueError, TypeError):
            return None
    
    def extract_coordinates_from_remarks(self, remarks: str) -> tuple:
        """Extract UTM coordinates from remarks string and convert to lat/long"""
        if not remarks or not isinstance(remarks, str):
            return None, None
            
        # Extract easting and northing from remarks
        easting_match = re.search(r'Easting:\s*([\d.]+)', remarks)
        northing_match = re.search(r'Northing:\s*([\d.]+)', remarks)
        
        if easting_match and northing_match:
            try:
                easting = float(easting_match.group(1))
                northing = float(northing_match.group(1))
                
                # Convert UTM Zone 32N to WGS84 lat/long
                # This is a simplified conversion - in production you'd use a proper library
                # For UTM Zone 32N (Norway), approximate conversion:
                # Latitude ≈ 59.0 + (northing - 74000) / 111000
                # Longitude ≈ 5.0 + (easting - 5400) / 70000
                
                latitude = 59.0 + (northing - 74000) / 111000
                longitude = 5.0 + (easting - 5400) / 70000
                
                return latitude, longitude
            except (ValueError, TypeError):
                return None, None
        
        return None, None

    def enhance_record_with_llm(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance record with additional canonical fields using hybrid LLM client"""
        if not self.llm_available or not self.llm_client:
            return record
            
        try:
            # Prepare context for LLM
            well_id = record.get('well_id', '')
            horizon_name = record.get('horizon_name', '')
            depth_start = record.get('depth_start', '')
            remarks = record.get('remarks', '')
            
            # Extract coordinates from remarks if available
            latitude, longitude = self.extract_coordinates_from_remarks(remarks)
            
            prompt = f"""Extract additional canonical fields from this well pick data and return as JSON:

Well ID: {well_id}
Formation: {horizon_name}
Depth: {depth_start}m
Remarks: {remarks}
Coordinates: Latitude={latitude}, Longitude={longitude} if available

Extract and return JSON with these additional fields:
- latitude, longitude (use provided coordinates or convert UTM Zone 32N from remarks to WGS84 lat/long)
- qc_flag (boolean, true if quality issues detected in remarks)
- country (from well naming pattern, likely Norway)
- field_name (from well naming, likely Volve)
- block_name (extract from well ID pattern)
- facies_code (geological facies classification from formation name)
- depth_end (estimate based on formation type and depth)

IMPORTANT:
- Use provided latitude/longitude if available, otherwise extract from remarks
- Do NOT include non-canonical fields like "formation", "depth", or "coordinates"
- Return only valid JSON with canonical field names

Return only valid JSON, no explanations."""
            
            # Use hybrid LLM client
            enhanced_data = self.llm_client.enhance_metadata(prompt, max_tokens=500)
            
            # enhanced_data is already parsed JSON from the LLM client
            enhanced_fields = enhanced_data
            
            # Filter out non-canonical fields and merge enhanced fields with original record
            canonical_fields = {
                'well_id', 'record_type', 'curve_name', 'depth_start', 'acquisition_date',
                'analyst', 'archie_a', 'archie_m', 'archie_n', 'bulk_vol_water', 'caliper',
                'carbonate_flag', 'clay_volume', 'coal_flag', 'core_permeability', 'core_porosity',
                'depth_end', 'facies_code', 'facies_confidence', 'file_checksum', 'file_origin',
                'formation_press', 'formation_temp', 'gas_oil_ratio', 'horizon_depth', 'horizon_name',
                'mud_flow_rate', 'mud_viscosity', 'mud_weight_actual', 'null_count', 'num_samples',
                'permeability', 'plan_azimuth', 'plan_inclination', 'plan_md', 'plan_tvd', 'porosity',
                'processing_date', 'processing_software', 'production_rate', 'pump_pressure',
                'qc_flag', 'remarks', 'resistivity_deep', 'resistivity_medium', 'resistivity_shallow',
                'rop', 'sample_interval', 'sample_max', 'sample_mean', 'sample_min', 'sample_stddev',
                'sand_flag', 'seismic_inline', 'seismic_sample_rate', 'seismic_trace_count',
                'seismic_xline', 'service_company', 'sp_curves', 'state_province', 'tool_type',
                'torque', 'version', 'vp', 'vs', 'vshale', 'water_resistivity', 'water_saturation',
                'weight_on_bit', 'country', 'field_name', 'block_name', 'latitude', 'longitude', 'elevation'
            }
            
            # Filter enhanced fields to only include canonical fields
            filtered_fields = {k: v for k, v in enhanced_fields.items() if k in canonical_fields}
            record.update(filtered_fields)
            
            self.logger.debug(f"Enhanced record for {well_id} with LLM")
            return record
            
        except Exception as e:
            self.logger.warning(f"LLM enhancement failed for {record.get('well_id', 'unknown')}: {e}")
            return record
    
    def parse_line(self, line: str, well_id: str) -> Optional[Dict[str, Any]]:
        """Parse a single data line and return canonical record"""
        # Skip empty lines and header lines
        if not line.strip() or line.startswith('Well ') or line.startswith('---') or line.startswith('#'):
            return None
            
        # Parse fixed-width format based on column positions
        # Format: Well_name(25) Surface_name(40) Obs#(5) Qlf(4) MD(10) TVD(10) TVDSS(10) TWT(10) Dip(7) Azi(7) Easting(10) Northing(10) Intrp(5)
        try:
            # Extract fields using fixed-width positions
            well_name = line[0:25].strip()
            surface_name = line[25:65].strip()
            obs_num = line[65:70].strip()
            qlf = line[70:74].strip()
            md = line[74:84].strip()
            tvd = line[84:94].strip()
            tvdss = line[94:104].strip()
            twt = line[104:114].strip()
            dip = line[114:121].strip()
            azi = line[121:128].strip()
            easting = line[128:138].strip()
            northing = line[138:148].strip()
            intrp = line[148:153].strip()
            
            # Parse numeric values
            md_val = self.parse_numeric_value(md)
            tvd_val = self.parse_numeric_value(tvd)
            dip_val = self.parse_numeric_value(dip)
            azi_val = self.parse_numeric_value(azi)
            easting_val = self.parse_numeric_value(easting)
            northing_val = self.parse_numeric_value(northing)
            
            # Create remarks
            remarks = self.create_remarks(qlf, twt, tvdss, obs_num, easting, northing)
            
            # Ensure remarks is always a string
            if isinstance(remarks, dict):
                remarks = json.dumps(remarks)
            
            # Create canonical record
            record = {
                "well_id": well_id,
                "record_type": "well_picks",
                "curve_name": "formation_top",
                "depth_start": md_val,
                "horizon_name": surface_name,
                "horizon_depth": md_val,
                "plan_md": md_val,
                "plan_tvd": tvd_val,
                "plan_azimuth": azi_val,
                "plan_inclination": dip_val,
                "remarks": remarks,
                "file_origin": os.path.basename(self.file_path),
                "processing_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Only return record if we have essential data
            if not well_id or not surface_name or md_val is None:
                return None
            
            # Remove None values
            record = {k: v for k, v in record.items() if v is not None}
            
            # Enhance with LLM if available (limit to first 5 records for testing)
            if self.llm_available and self.record_count < 5:
                record = self.enhance_record_with_llm(record)
                self.record_count += 1
            
            return record
            
        except Exception as e:
            self.logger.warning(f"Error parsing line '{line.strip()}': {e}")
            return None
    
    def parse(self) -> Dict[str, Any]:
        """Parse the well picks .dat file and return canonical format"""
        try:
            self.logger.info(f"Parsing well picks file: {self.file_path}")
            
            # Validate file
            if not self.validate_file():
                return self.create_error_result("File validation failed")
            
            records = []
            current_well_id = None
            line_count = 0
            
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_count += 1
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Check if this is a well header line
                    if line.startswith('Well ') and not line.startswith('Well name'):
                        # Extract well ID from header
                        well_match = re.search(r'Well\s+(.+)', line)
                        if well_match:
                            current_well_id = self.extract_well_id(well_match.group(1))
                            self.logger.info(f"Found well: {current_well_id}")
                        continue
                    
                    # Skip separator lines
                    if line.startswith('---'):
                        continue
                    
                    # Skip header lines (column names)
                    if 'Well name' in line and 'Surface name' in line:
                        continue
                    
                    # Parse data line
                    if current_well_id:
                        record = self.parse_line(line, current_well_id)
                        if record:
                            records.append(record)
            
            self.logger.info(f"Parsed {len(records)} formation top records from {line_count} lines")
            
            # Create metadata
            metadata = {
                "parser_name": "WellPicksParser",
                "file_path": self.file_path,
                "file_size_bytes": os.path.getsize(self.file_path),
                "processing_timestamp": datetime.utcnow().isoformat(),
                "total_records": len(records),
                "wells_found": len(set(r["well_id"] for r in records)),
                "formations_found": len(set(r["horizon_name"] for r in records if r.get("horizon_name")))
            }
            
            return {
                "metadata": metadata,
                "records": records
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing well picks file: {e}")
            return self.create_error_result(f"Parsing failed: {str(e)}")

def main():
    """Test the parser with a sample file"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python well_picks_parser.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create parser and parse file
    parser = DatParser(file_path)
    result = parser.parse()
    
    # Print results
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)
    
    print(f"✅ Successfully parsed {result['metadata']['total_records']} records")
    print(f"📊 Wells found: {result['metadata']['wells_found']}")
    print(f"🏔️  Formations found: {result['metadata']['formations_found']}")
    
    # Print first few records as example
    if result['records']:
        print(f"\n📋 Sample records:")
        for i, record in enumerate(result['records'][:3]):
            print(f"  Record {i+1}: {record['well_id']} - {record['horizon_name']} at {record['depth_start']}m")

if __name__ == "__main__":
    main()
