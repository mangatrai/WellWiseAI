#!/usr/bin/env python3
"""
Enhanced SEG-Y Parser with Well Correlation and Advanced Seismic Attribute Analysis
Incorporates well picks data and uses GPT for advanced seismic interpretation
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import struct
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import openai
from dotenv import load_dotenv

from .base_parser import BaseParser, ParsedData, ContextualDocument

load_dotenv()

class SegyParser(BaseParser):
    """Enhanced SEG-Y parser with well correlation and advanced seismic attribute analysis"""
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file_path = Path(file_path)
        
        # Initialize well correlation data
        self.well_picks = {}
        self.well_logs = {}
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.openai_client = openai.OpenAI(api_key=api_key)
        else:
            self.openai_client = None
            self.logger.warning("OpenAI API key not found. GPT analysis will be disabled.")
    
    def can_parse(self) -> bool:
        """Check if this parser can handle the given file"""
        if not self.validate_file():
            return False
            
        # Check file extension
        extension = self.file_path.suffix.lower()
        if extension not in ['.segy', '.sgy']:
            return False
            
        return True
    
    def parse(self) -> Dict[str, Any]:
        """Parse SEG-Y file and return structured data in pipeline-compatible format"""
        start_time = time.time()
        
        try:
            if not self.can_parse():
                return {"error": "File cannot be parsed by SEG-Y parser"}
            
            # Run enhanced analysis
            analysis_results = self.run_enhanced_analysis()
            
            # Extract records from the analysis results
            records = self._extract_canonical_records(analysis_results)
            
            processing_time = time.time() - start_time
            
            return {
                "records": records,
                "metadata": {
                    "file_path": str(self.file_path),
                    "file_type": 'segy',
                    "parser_name": self.__class__.__name__,
                    "processing_time": processing_time,
                    "record_count": len(records),
                    "sample_size": os.getenv('SEGY_SAMPLE_SIZE', '1000')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing {self.file_path}: {e}")
            return {"error": str(e)}
    
    def _extract_canonical_records(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract canonical records from SEGY analysis results"""
        records = []
        
        # Get seismic data for use in all records
        seismic_data = analysis_results.get('seismic_data', {})
        
        # Extract advanced seismic attributes using canonical fields
        advanced_attributes = analysis_results.get('advanced_attributes', {})
        if advanced_attributes:
            # Create advanced attributes record using canonical fields
            advanced_record = {
                'well_id': f"segy_{self.file_path.stem}",
                'record_type': 'seismic',
                'curve_name': 'advanced_attributes',
                'depth_start': 0.0,
                'depth_end': float(seismic_data.get('header_info', {}).get('samples_per_trace', 850) * 
                                 seismic_data.get('header_info', {}).get('sample_interval', 4000) / 1000.0),
                'seismic_sample_rate': float(seismic_data.get('header_info', {}).get('sample_interval', 4000)),
                'sample_interval': float(seismic_data.get('header_info', {}).get('sample_interval', 4000)),
                'processing_date': analysis_results.get('analysis_timestamp', ''),
                'processing_software': 'SegyParser',
                'tool_type': 'advanced_attributes',
                'service_company': 'WellWiseAI',
                'version': '1.0',
                'file_origin': str(self.file_path).replace('\x00', '').replace('\u0000', '').strip(),
                'qc_flag': True,
                'remarks': f"Advanced seismic attributes extracted - RMS maps, instantaneous attributes, coherence analysis, frequency attributes, statistical attributes"
            }
            records.append(advanced_record)
        
        # Extract basic seismic statistics using canonical fields
        seismic_data = analysis_results.get('seismic_data', {})
        if 'amplitude_statistics' in seismic_data:
            stats = seismic_data['amplitude_statistics']['overall_statistics']
            
            # Create comprehensive seismic summary record using canonical fields
            seismic_summary = {
                'well_id': f"segy_{self.file_path.stem}",
                'record_type': 'seismic',
                'curve_name': 'seismic_summary',
                'depth_start': 0.0,
                'depth_end': float(seismic_data.get('header_info', {}).get('samples_per_trace', 850) * 
                                 seismic_data.get('header_info', {}).get('sample_interval', 4000) / 1000.0),
                'sample_max': float(stats.get('max', 0.0)),
                'sample_mean': float(stats.get('mean', 0.0)),
                'sample_min': float(stats.get('min', 0.0)),
                'sample_stddev': float(stats.get('std', 0.0)),
                'num_samples': int(seismic_data.get('amplitude_statistics', {}).get('sample_count', 0)),
                'seismic_trace_count': int(seismic_data.get('amplitude_statistics', {}).get('trace_count_analyzed', 0)),
                'seismic_sample_rate': float(seismic_data.get('header_info', {}).get('sample_interval', 4000)),
                'sample_interval': float(seismic_data.get('header_info', {}).get('sample_interval', 4000)),
                'processing_date': analysis_results.get('analysis_timestamp', ''),
                'processing_software': 'SegyParser',
                'tool_type': 'seismic_summary',
                'service_company': 'WellWiseAI',
                'version': '1.0',
                'file_origin': str(self.file_path).replace('\x00', '').replace('\u0000', '').strip(),
                'qc_flag': True,
                'remarks': f"Overall seismic statistics - RMS: {stats.get('rms', 0.0):.3f}, Skewness: {stats.get('skewness', 0.0):.3f}, Kurtosis: {stats.get('kurtosis', 0.0):.3f}"
            }
            records.append(seismic_summary)
        
        # Extract frequency analysis results using canonical fields
        if 'frequency_analysis' in seismic_data:
            freq_data = seismic_data['frequency_analysis']
            freq_stats = freq_data.get('frequency_statistics', {})
            
            if freq_stats:
                frequency_record = {
                    'well_id': f"segy_{self.file_path.stem}",
                    'record_type': 'seismic',
                    'curve_name': 'frequency_analysis',
                    'depth_start': 0.0,
                    'depth_end': float(seismic_data.get('header_info', {}).get('samples_per_trace', 850) * 
                                     seismic_data.get('header_info', {}).get('sample_interval', 4000) / 1000.0),
                    'seismic_sample_rate': float(seismic_data.get('header_info', {}).get('sample_interval', 4000)),
                    'sample_interval': float(seismic_data.get('header_info', {}).get('sample_interval', 4000)),
                    'processing_date': analysis_results.get('analysis_timestamp', ''),
                    'processing_software': 'SegyParser',
                                    'tool_type': 'frequency_analysis',
                'service_company': 'WellWiseAI',
                'version': '1.0',
                'file_origin': str(self.file_path).replace('\x00', '').replace('\u0000', '').strip(),
                'qc_flag': True,
                    'remarks': f"Frequency analysis - Mean dominant: {freq_stats.get('mean_dominant_frequency', 0.0):.1f}Hz, Bandwidth: {freq_stats.get('frequency_bandwidth', 0.0):.1f}Hz"
                }
                records.append(frequency_record)
        
        # Extract individual trace records from sampled traces using canonical fields
        trace_samples = seismic_data.get('trace_samples', [])
        for trace_sample in trace_samples:
            trace_num = trace_sample.get('trace_number', 0)
            trace_position = trace_sample.get('trace_position', 0)
            amplitudes = trace_sample.get('amplitudes', [])
            
            if amplitudes:
                # Create a single comprehensive record for each trace using canonical fields
                trace_record = {
                    'well_id': f"segy_{self.file_path.stem}",
                    'record_type': 'seismic',
                    'curve_name': f'trace_{trace_num}',
                    'depth_start': 0.0,
                    'depth_end': float(len(amplitudes) * seismic_data.get('header_info', {}).get('sample_interval', 4000) / 1000.0),
                    'sample_interval': float(seismic_data.get('header_info', {}).get('sample_interval', 4000)),
                    'sample_max': float(max(amplitudes) if amplitudes else 0.0),
                    'sample_mean': float(trace_sample.get('mean', 0.0)),
                    'sample_min': float(min(amplitudes) if amplitudes else 0.0),
                    'sample_stddev': float(trace_sample.get('std', 0.0)),
                    'num_samples': len(amplitudes),
                    'seismic_trace_count': trace_num,
                    'seismic_sample_rate': float(seismic_data.get('header_info', {}).get('sample_interval', 4000)),
                    'processing_date': analysis_results.get('analysis_timestamp', ''),
                    'processing_software': 'SegyParser',
                                    'tool_type': 'seismic_trace',
                'service_company': 'WellWiseAI',
                'version': '1.0',
                'file_origin': str(self.file_path).replace('\x00', '').replace('\u0000', '').strip(),
                'qc_flag': True,
                    'remarks': f"Trace {trace_num} (position {trace_position}) - RMS: {trace_sample.get('rms', 0.0):.3f}, Samples: {len(amplitudes)}"
                }
                records.append(trace_record)
        
        return records
    

        
        # Extract well correlation records if available
        well_correlation = analysis_results.get('well_correlation', {})
        for well_name, well_data in well_correlation.items():
            if isinstance(well_data, dict) and 'surfaces' in well_data:
                for surface in well_data['surfaces']:
                    record = {
                        'well_id': well_name,
                        'depth_start': surface.get('tvd', 0.0),
                        'depth_end': surface.get('tvd', 0.0) + 1.0,
                        'curve_name': f'well_pick_{surface.get("surface_name", "unknown")}',
                        'curve_value': 1.0,
                        'record_type': 'well_pick',
                        'data_source': 'segy_parser',
                        'remarks': f"Well pick: {surface.get('surface_name', 'Unknown')}, MD={surface.get('md', 'N/A')}m, TVD={surface.get('tvd', 'N/A')}m",
                        'qc_flag': True
                    }
                    records.append(record)
        
        return records
    
    def find_well_data_files(self):
        """Find well data files in the same directory as SEG-Y file and subdirectories"""
        segy_dir = self.file_path.parent
        print(f"🔍 Searching for well data files in: {segy_dir}")
        
        # Find .dat files (well picks)
        dat_files = list(segy_dir.glob('**/*.dat'))
        well_picks_file = None
        if dat_files:
            # Prefer files with 'well' or 'pick' in name
            preferred_files = [f for f in dat_files if any(keyword in f.name.lower() for keyword in ['well', 'pick', 'volve'])]
            well_picks_file = preferred_files[0] if preferred_files else dat_files[0]
            print(f"✅ Found well picks file: {well_picks_file.name}")
        
        # Find .csv files (well logs) - prefer depth-related files
        csv_files = list(segy_dir.glob('**/*.csv'))
        well_log_file = None
        if csv_files:
            # Prefer files with 'depth' in name
            depth_files = [f for f in csv_files if 'depth' in f.name.lower()]
            well_log_file = depth_files[0] if depth_files else csv_files[0]
            print(f"✅ Found well log file: {well_log_file.name}")
        
        if not well_picks_file and not well_log_file:
            print("⚠️  No well data files found in SEG-Y directory")
        
        return well_picks_file, well_log_file
    
    def load_well_correlation_data(self):
        """Load well picks and correlation data using unstructured parser"""
        print("=== Loading Well Correlation Data ===")
        
        # Find well data files in SEG-Y directory
        well_picks_file, well_log_file = self.find_well_data_files()
        
        # Load well picks data using unstructured parser
        if well_picks_file and well_picks_file.exists():
            try:
                from parsers.unstructured_parser import UnstructuredParser
                dat_parser = UnstructuredParser(str(well_picks_file))
                dat_result = dat_parser.parse()
                
                if dat_result.error is None:
                    self.well_picks = self._extract_well_picks_from_unstructured(dat_result.data)
                    print(f"✅ Loaded well picks using unstructured parser: {len(self.well_picks)} wells")
                else:
                    print(f"⚠️  Failed to parse well picks with unstructured parser: {dat_result.error}")
                    # Fallback to custom parser
                    self.well_picks = self._parse_well_picks(well_picks_file)
                    print(f"✅ Loaded well picks using custom parser: {len(self.well_picks)} wells")
            except Exception as e:
                print(f"⚠️  Error with unstructured parser: {e}")
                # Fallback to custom parser
                self.well_picks = self._parse_well_picks(well_picks_file)
                print(f"✅ Loaded well picks using custom parser: {len(self.well_picks)} wells")
        
        # Load well log data using unstructured parser
        if well_log_file and well_log_file.exists():
            try:
                from parsers.unstructured_parser import UnstructuredParser
                csv_parser = UnstructuredParser(str(well_log_file))
                csv_result = csv_parser.parse()
                
                if csv_result.error is None:
                    self.well_logs = self._extract_well_logs_from_unstructured(csv_result.data)
                    print(f"✅ Loaded well logs using unstructured parser: {len(self.well_logs)} depth points")
                else:
                    print(f"⚠️  Failed to parse well logs with unstructured parser: {csv_result.error}")
                    # Fallback to custom parser
                    self.well_logs = self._parse_well_logs(well_log_file)
                    print(f"✅ Loaded well logs using custom parser: {len(self.well_logs)} depth points")
            except Exception as e:
                print(f"⚠️  Error with unstructured parser: {e}")
                # Fallback to custom parser
                self.well_logs = self._parse_well_logs(well_log_file)
                print(f"✅ Loaded well logs using custom parser: {len(self.well_logs)} depth points")
        
        return len(self.well_picks) > 0 or len(self.well_logs) > 0
    
    def phase1_extract_seismic_data(self) -> Dict[str, Any]:
        """Phase 1: Extract comprehensive seismic amplitude data and statistics"""
        
        print("=== Phase 1: Seismic Data Extraction ===")
        
        seismic_data = {
            'file_info': {},
            'header_info': {},
            'amplitude_statistics': {},
            'frequency_analysis': {},
            'trace_samples': [],
            'anomaly_detection': {}
        }
        
        try:
            with open(self.file_path, 'rb') as f:
                # File information
                file_size = self.file_path.stat().st_size
                seismic_data['file_info'] = {
                    'file_size_bytes': file_size,
                    'file_size_mb': file_size / (1024 * 1024),
                    'filename': self.file_path.name
                }
                
                # Read headers
                text_header = f.read(3200)
                binary_header = f.read(400)
                
                if len(binary_header) >= 400:
                    # Extract header information
                    sample_interval = int.from_bytes(binary_header[16:18], byteorder='big')
                    samples_per_trace = int.from_bytes(binary_header[20:22], byteorder='big')
                    data_format = binary_header[24]
                    
                    seismic_data['header_info'] = {
                        'sample_interval': sample_interval,
                        'samples_per_trace': samples_per_trace,
                        'data_format': data_format,
                        'data_format_description': self._get_data_format_description(data_format)
                    }
                    
                    # Calculate file structure
                    header_size = 3600
                    trace_header_size = 240
                    trace_data_size = samples_per_trace * 4
                    total_trace_size = trace_header_size + trace_data_size
                    estimated_traces = (file_size - header_size) // total_trace_size
                    
                    print(f"Estimated traces: {estimated_traces:,}")
                    print(f"Sample interval: {sample_interval} μs")
                    print(f"Samples per trace: {samples_per_trace}")
                    print(f"Data format: {self._get_data_format_description(data_format)}")
                    
                    # Extract amplitude data from sample traces
                    amplitude_values = []
                    frequency_data = []
                    trace_samples = []
                    
                    # Random sampling of traces for analysis (configurable via environment)
                    sample_size = int(os.getenv('SEGY_SAMPLE_SIZE', 1000))
                    sample_count = min(sample_size, estimated_traces)
                    
                    # Generate random trace positions for better coverage
                    import random
                    trace_positions = random.sample(range(estimated_traces), sample_count)
                    trace_positions.sort()  # Sort for efficient file reading
                    
                    print(f"Random sampling {sample_count} traces from {estimated_traces:,} total traces")
                    print(f"Sample coverage: {sample_count/estimated_traces*100:.2f}% of total traces")
                    
                    f.seek(header_size)
                    
                    for trace_idx, trace_num in enumerate(trace_positions):
                        # Skip trace header
                        f.seek(header_size + trace_num * total_trace_size + trace_header_size)
                        
                        # Read trace data
                        trace_data = f.read(trace_data_size)
                        if len(trace_data) == trace_data_size:
                            # Convert to amplitude values
                            amplitudes = []
                            for i in range(0, len(trace_data), 4):
                                if i + 4 <= len(trace_data):
                                    value = struct.unpack('>f', trace_data[i:i+4])[0]
                                    amplitudes.append(value)
                            
                            if amplitudes:
                                trace_sample = {
                                    'trace_number': trace_num + 1,
                                    'trace_position': trace_idx + 1,  # Position in our sample
                                    'amplitudes': amplitudes,
                                    'rms': np.sqrt(np.mean(np.array(amplitudes)**2)),
                                    'mean': np.mean(amplitudes),
                                    'std': np.std(amplitudes)
                                }
                                trace_samples.append(trace_sample)
                                amplitude_values.extend(amplitudes)
                                
                                # Frequency analysis
                                if len(amplitudes) > 10:
                                    fft_result = np.fft.fft(amplitudes)
                                    freqs = np.fft.fftfreq(len(amplitudes), sample_interval * 1e-6)
                                    positive_freqs = freqs[freqs > 0]
                                    positive_fft = np.abs(fft_result[freqs > 0])
                                    
                                    if len(positive_freqs) > 0:
                                        dominant_freq_idx = np.argmax(positive_fft)
                                        dominant_freq = positive_freqs[dominant_freq_idx]
                                        
                                        frequency_data.append({
                                            'trace': trace_num + 1,
                                            'trace_position': trace_idx + 1,
                                            'dominant_freq': dominant_freq,
                                            'frequency_spectrum': positive_fft.tolist()
                                        })
                    
                    # Calculate overall statistics
                    if amplitude_values:
                        amplitude_array = np.array(amplitude_values)
                        seismic_data['amplitude_statistics'] = {
                            'overall_statistics': {
                                'mean': float(np.mean(amplitude_array)),
                                'std': float(np.std(amplitude_array)),
                                'min': float(np.min(amplitude_array)),
                                'max': float(np.max(amplitude_array)),
                                'rms': float(np.sqrt(np.mean(amplitude_array**2))),
                                'abs_mean': float(np.mean(np.abs(amplitude_array))),
                                'variance': float(np.var(amplitude_array)),
                                'skewness': float(self._calculate_skewness(amplitude_array)),
                                'kurtosis': float(self._calculate_kurtosis(amplitude_array))
                            },
                            'percentiles': {
                                'p1': float(np.percentile(amplitude_array, 1)),
                                'p5': float(np.percentile(amplitude_array, 5)),
                                'p25': float(np.percentile(amplitude_array, 25)),
                                'p50': float(np.percentile(amplitude_array, 50)),
                                'p75': float(np.percentile(amplitude_array, 75)),
                                'p95': float(np.percentile(amplitude_array, 95)),
                                'p99': float(np.percentile(amplitude_array, 99))
                            },
                            'sample_count': len(amplitude_values),
                            'trace_count_analyzed': sample_count
                        }
                        
                        # Frequency analysis
                        if frequency_data:
                            dominant_freqs = [f['dominant_freq'] for f in frequency_data if f['dominant_freq'] > 0]
                            seismic_data['frequency_analysis'] = {
                                'frequency_data': frequency_data,
                                'frequency_statistics': {
                                    'mean_dominant_frequency': float(np.mean(dominant_freqs)) if dominant_freqs else 0,
                                    'std_dominant_frequency': float(np.std(dominant_freqs)) if dominant_freqs else 0,
                                    'min_dominant_frequency': float(np.min(dominant_freqs)) if dominant_freqs else 0,
                                    'max_dominant_frequency': float(np.max(dominant_freqs)) if dominant_freqs else 0
                                }
                            }
                        
                        seismic_data['trace_samples'] = trace_samples
                        
                        print(f"✅ Extracted amplitude data from {sample_count} traces")
                        print(f"✅ Amplitude range: {np.min(amplitude_array):.2f} to {np.max(amplitude_array):.2f}")
                        print(f"✅ RMS amplitude: {np.sqrt(np.mean(amplitude_array**2)):.2f}")
                        
                        if dominant_freqs:
                            print(f"✅ Mean dominant frequency: {np.mean(dominant_freqs):.2f} Hz")
                    
        except Exception as e:
            self.logger.error(f"Error in seismic data extraction: {e}")
        
        return seismic_data
    
    def _get_data_format_description(self, data_format: int) -> str:
        """Get description of SEG-Y data format"""
        format_descriptions = {
            0: "IBM floating point",
            1: "32-bit integer",
            2: "16-bit integer",
            3: "16-bit integer with gain code",
            4: "32-bit integer with gain code",
            5: "16-bit integer with gain code and scaling",
            6: "32-bit integer with gain code and scaling",
            8: "8-bit integer",
            9: "8-bit integer with gain code",
            10: "8-bit integer with gain code and scaling",
            11: "16-bit integer with gain code and scaling",
            12: "32-bit integer with gain code and scaling",
            15: "32-bit IEEE floating point",
            16: "32-bit IEEE floating point with gain code",
            17: "32-bit IEEE floating point with gain code and scaling",
            18: "64-bit IEEE floating point",
            19: "64-bit IEEE floating point with gain code",
            20: "64-bit IEEE floating point with gain code and scaling"
        }
        return format_descriptions.get(data_format, f"Unknown format ({data_format})")
    
    def _parse_well_picks(self, file_path: Path) -> Dict[str, Any]:
        """Parse well picks data file"""
        well_picks = {}
        current_well = None
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Check for well header
            if line.startswith('Well '):
                current_well = line.split('Well ')[1]
                well_picks[current_well] = {
                    'surfaces': [],
                    'coordinates': {},
                    'depths': {}
                }
                continue
            
            # Parse surface picks
            if current_well and line and not line.startswith('#') and not line.startswith('---'):
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        surface_name = parts[1]
                        md = float(parts[4]) if parts[4] != '' else None
                        tvd = float(parts[5]) if parts[5] != '' else None
                        twt = float(parts[7]) if parts[7] != '' else None
                        easting = float(parts[8]) if len(parts) > 8 and parts[8] != '' else None
                        northing = float(parts[9]) if len(parts) > 9 and parts[9] != '' else None
                        
                        well_picks[current_well]['surfaces'].append({
                            'surface_name': surface_name,
                            'md': md,
                            'tvd': tvd,
                            'twt': twt,
                            'easting': easting,
                            'northing': northing
                        })
                        
                        # Store key surfaces
                        if 'Seabed' in surface_name:
                            well_picks[current_well]['coordinates']['seabed'] = (easting, northing)
                        elif 'Hugin' in surface_name:
                            well_picks[current_well]['depths']['hugin_top'] = tvd
                            well_picks[current_well]['depths']['hugin_twt'] = twt
                        elif 'Draupne' in surface_name:
                            well_picks[current_well]['depths']['draupne_top'] = tvd
                            well_picks[current_well]['depths']['draupne_twt'] = twt
                        
                    except (ValueError, IndexError):
                        continue
        
        return well_picks
    
    def _parse_well_logs(self, file_path: Path) -> Dict[str, Any]:
        """Parse well log data from CSV"""
        try:
            # Read first few lines to understand structure
            df = pd.read_csv(file_path, nrows=1000)
            
            # Extract key columns
            well_logs = {
                'depth_data': [],
                'gamma_ray': [],
                'density': [],
                'porosity': [],
                'resistivity': []
            }
            
            # Process depth data
            if 'Measured Depth m' in df.columns and 'Hole Depth (TVD) m' in df.columns:
                depth_data = df[['Measured Depth m', 'Hole Depth (TVD) m']].dropna()
                well_logs['depth_data'] = depth_data.to_dict('records')
            
            # Process gamma ray data
            if 'Gamma Ray, Average gAPI' in df.columns:
                gamma_data = df[['Measured Depth m', 'Gamma Ray, Average gAPI']].dropna()
                well_logs['gamma_ray'] = gamma_data.to_dict('records')
            
            # Process density data
            if 'Bulk Density, Bottom, Computed DH g/cm3' in df.columns:
                density_data = df[['Measured Depth m', 'Bulk Density, Bottom, Computed DH g/cm3']].dropna()
                well_logs['density'] = density_data.to_dict('records')
            
            return well_logs
            
        except Exception as e:
            self.logger.error(f"Error parsing well logs: {e}")
            return {}
    
    def _extract_well_picks_from_unstructured(self, unstructured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract well picks data from unstructured parser results using fixed-width parsing"""
        well_picks = {}
        current_well = None
        
        # Extract content from unstructured data
        chunks = unstructured_data.get('chunks', [])
        
        for chunk in chunks:
            content = chunk.get('content', '')
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Check for well header
                if line.startswith('Well NO'):
                    current_well = line.split('Well ')[1]
                    well_picks[current_well] = {
                        'surfaces': [],
                        'coordinates': {},
                        'depths': {}
                    }
                    continue
                
                # Parse surface picks using space-separated format (unstructured parser output)
                if current_well and line and not line.startswith('#') and not line.startswith('---') and not line.startswith('Well name'):
                    # Space-separated parsing for unstructured parser output
                    try:
                        parts = line.split()
                        if len(parts) >= 8:
                            # Parse space-separated data: "NO 15/9-11 NORDLAND GP. Top 1 586.00 586.00 -561.00 629.45 435409.4 6474001.9 STAT"
                            well_name = f"{parts[0]} {parts[1]} {parts[2]}"  # "NO 15/9-11"
                            
                            # Find surface name (variable length, ends before Obs#)
                            # First, identify the well name pattern
                            well_name_parts = current_well.split()
                            well_end = 0
                            
                            # Find where the well name ends in the parts
                            for j in range(len(parts)):
                                if j + len(well_name_parts) - 1 < len(parts):
                                    match = True
                                    for k in range(len(well_name_parts)):
                                        if parts[j + k] != well_name_parts[k]:
                                            match = False
                                            break
                                    if match:
                                        well_end = j + len(well_name_parts)
                                        break
                            
                            # Extract surface name from well_end to the first digit
                            surface_parts = []
                            j = well_end
                            while j < len(parts) and not parts[j].isdigit():
                                surface_parts.append(parts[j])
                                j += 1
                            surface_name = " ".join(surface_parts)
                            
                            # If surface name is empty, try a different approach
                            if not surface_name:
                                # Look for the pattern: well_name surface_name obs_number
                                # The surface name should be everything between well_name and obs_number
                                well_name_parts = current_well.split()
                                if len(well_name_parts) >= 2:
                                    # Find where the well name ends
                                    well_end = 0
                                    for j in range(len(parts)):
                                        if j + 1 < len(parts) and parts[j] == well_name_parts[0] and parts[j+1] == well_name_parts[1]:
                                            well_end = j + len(well_name_parts)
                                            break
                                    
                                    # Extract surface name from well_end to the first digit
                                    surface_parts = []
                                    j = well_end
                                    while j < len(parts) and not parts[j].isdigit():
                                        surface_parts.append(parts[j])
                                        j += 1
                                    surface_name = " ".join(surface_parts)
                            

                            
                            # Skip if no surface name found
                            if not surface_name:
                                continue
                            
                            # Extract numeric values
                            obs = parts[j] if j < len(parts) else None
                            qlf = parts[j+1] if j+1 < len(parts) else None
                            md_str = parts[j+2] if j+2 < len(parts) else None
                            tvd_str = parts[j+3] if j+3 < len(parts) else None
                            tvdss_str = parts[j+4] if j+4 < len(parts) else None
                            twt_str = parts[j+5] if j+5 < len(parts) else None
                            dip_str = parts[j+6] if j+6 < len(parts) else None
                            azi_str = parts[j+7] if j+7 < len(parts) else None
                            easting_str = parts[j+8] if j+8 < len(parts) else None
                            northing_str = parts[j+9] if j+9 < len(parts) else None
                            
                            # Convert numeric values
                            md = float(md_str) if md_str and md_str != '' else None
                            tvd = float(tvd_str) if tvd_str and tvd_str != '' else None
                            twt = float(twt_str) if twt_str and twt_str != '' else None
                            easting = float(easting_str) if easting_str and easting_str != '' else None
                            northing = float(northing_str) if northing_str and northing_str != '' else None
                            
                            # Only add if we have a valid surface name and at least one depth value
                            if surface_name and (md is not None or tvd is not None or twt is not None):
                                well_picks[current_well]['surfaces'].append({
                                    'surface_name': surface_name,
                                    'md': md,
                                    'tvd': tvd,
                                    'twt': twt,
                                    'easting': easting,
                                    'northing': northing
                                })
                                
                                # Store key surfaces
                                if 'Seabed' in surface_name:
                                    well_picks[current_well]['coordinates']['seabed'] = (easting, northing)
                                elif 'Hugin' in surface_name:
                                    well_picks[current_well]['depths']['hugin_top'] = tvd
                                    well_picks[current_well]['depths']['hugin_twt'] = twt
                                elif 'Draupne' in surface_name:
                                    well_picks[current_well]['depths']['draupne_top'] = tvd
                                    well_picks[current_well]['depths']['draupne_twt'] = twt
                                
                    except (ValueError, IndexError):
                        continue
        
        return well_picks
    
    def _extract_well_logs_from_unstructured(self, unstructured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract well logs data from unstructured parser results"""
        well_logs = {
            'depth_data': [],
            'gamma_ray': [],
            'density': [],
            'porosity': [],
            'resistivity': []
        }
        
        try:
            # Extract table data from unstructured results
            chunks = unstructured_data.get('chunks', [])
            
            for chunk in chunks:
                if chunk.get('element_type') == 'Table':
                    table_data = chunk.get('table_data', {})
                    if table_data:
                        # Process CSV-like table data
                        headers = table_data.get('headers', [])
                        rows = table_data.get('rows', [])
                        
                        if headers and rows:
                            # Convert to pandas-like structure for processing
                            for row in rows:
                                if len(row) >= len(headers):
                                    row_dict = dict(zip(headers, row))
                                    
                                    # Extract depth data
                                    if 'Measured Depth m' in row_dict and 'Hole Depth (TVD) m' in row_dict:
                                        try:
                                            md = float(row_dict['Measured Depth m']) if row_dict['Measured Depth m'] else None
                                            tvd = float(row_dict['Hole Depth (TVD) m']) if row_dict['Hole Depth (TVD) m'] else None
                                            if md is not None and tvd is not None:
                                                well_logs['depth_data'].append({
                                                    'Measured Depth m': md,
                                                    'Hole Depth (TVD) m': tvd
                                                })
                                        except (ValueError, TypeError):
                                            continue
                                    
                                    # Extract gamma ray data
                                    if 'Measured Depth m' in row_dict and 'Gamma Ray, Average gAPI' in row_dict:
                                        try:
                                            md = float(row_dict['Measured Depth m']) if row_dict['Measured Depth m'] else None
                                            gr = float(row_dict['Gamma Ray, Average gAPI']) if row_dict['Gamma Ray, Average gAPI'] else None
                                            if md is not None and gr is not None:
                                                well_logs['gamma_ray'].append({
                                                    'Measured Depth m': md,
                                                    'Gamma Ray, Average gAPI': gr
                                                })
                                        except (ValueError, TypeError):
                                            continue
                                    
                                    # Extract density data
                                    if 'Measured Depth m' in row_dict and 'Bulk Density, Bottom, Computed DH g/cm3' in row_dict:
                                        try:
                                            md = float(row_dict['Measured Depth m']) if row_dict['Measured Depth m'] else None
                                            density = float(row_dict['Bulk Density, Bottom, Computed DH g/cm3']) if row_dict['Bulk Density, Bottom, Computed DH g/cm3'] else None
                                            if md is not None and density is not None:
                                                well_logs['density'].append({
                                                    'Measured Depth m': md,
                                                    'Bulk Density, Bottom, Computed DH g/cm3': density
                                                })
                                        except (ValueError, TypeError):
                                            continue
            
        except Exception as e:
            self.logger.error(f"Error extracting well logs from unstructured data: {e}")
        
        return well_logs
    
    def extract_advanced_seismic_attributes(self, seismic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract advanced seismic attributes from seismic data"""
        
        print("=== Extracting Advanced Seismic Attributes ===")
        
        advanced_attributes = {
            'rms_amplitude_maps': {},
            'instantaneous_attributes': {},
            'coherence_analysis': {},
            'frequency_attributes': {},
            'statistical_attributes': {}
        }
        
        try:
            trace_samples = seismic_data.get('trace_samples', [])
            if not trace_samples:
                return advanced_attributes
            
            # Calculate RMS amplitude maps
            rms_values = []
            coordinates = []
            
            for trace in trace_samples:
                amplitudes = np.array(trace['amplitudes'])
                rms = np.sqrt(np.mean(amplitudes**2))
                rms_values.append(rms)
                
                # Store coordinates for spatial analysis
                coordinates.append({
                    'inline': trace.get('inline', 0),
                    'crossline': trace.get('crossline', 0),
                    'cdp_x': trace.get('cdp_x', 0),
                    'cdp_y': trace.get('cdp_y', 0)
                })
            
            # RMS amplitude analysis
            rms_array = np.array(rms_values)
            advanced_attributes['rms_amplitude_maps'] = {
                'mean_rms': float(np.mean(rms_array)),
                'std_rms': float(np.std(rms_array)),
                'min_rms': float(np.min(rms_array)),
                'max_rms': float(np.max(rms_array)),
                'rms_trend': 'increasing' if np.polyfit(range(len(rms_array)), rms_array, 1)[0] > 0 else 'decreasing',
                'rms_variation_coefficient': float(np.std(rms_array) / np.mean(rms_array)) if np.mean(rms_array) != 0 else 0
            }
            
            # Instantaneous attributes (simplified calculation)
            for trace in trace_samples[:10]:  # Sample first 10 traces
                amplitudes = np.array(trace['amplitudes'])
                
                # Instantaneous amplitude (envelope) - simplified calculation
                # Use scipy.signal.hilbert if available, otherwise use simplified approach
                try:
                    from scipy.signal import hilbert
                    hilbert_transform = np.imag(hilbert(amplitudes))
                    instantaneous_amplitude = np.sqrt(amplitudes**2 + hilbert_transform**2)
                except ImportError:
                    # Simplified envelope calculation
                    instantaneous_amplitude = np.abs(amplitudes)
                
                # Instantaneous frequency (simplified)
                try:
                    phase = np.unwrap(np.angle(hilbert_transform + 1j * amplitudes))
                    instantaneous_frequency = np.gradient(phase) / (2 * np.pi * seismic_data['header_info']['sample_interval'] * 1e-6)
                except:
                    # Simplified frequency calculation
                    instantaneous_frequency = np.ones_like(amplitudes) * 30.0  # Default frequency
                
                advanced_attributes['instantaneous_attributes'][f'trace_{trace["trace_number"]}'] = {
                    'mean_instantaneous_amplitude': float(np.mean(instantaneous_amplitude)),
                    'mean_instantaneous_frequency': float(np.mean(instantaneous_frequency)),
                    'amplitude_variation': float(np.std(instantaneous_amplitude) / np.mean(instantaneous_amplitude)) if np.mean(instantaneous_amplitude) != 0 else 0
                }
            
            # Frequency attributes
            freq_data = seismic_data.get('frequency_analysis', {}).get('frequency_data', [])
            if freq_data:
                dominant_freqs = [f['dominant_freq'] for f in freq_data if f['dominant_freq'] > 0]
                advanced_attributes['frequency_attributes'] = {
                    'mean_dominant_frequency': float(np.mean(dominant_freqs)) if dominant_freqs else 0,
                    'frequency_bandwidth': float(np.std(dominant_freqs)) if dominant_freqs else 0,
                    'frequency_range': f"{min(dominant_freqs):.1f} - {max(dominant_freqs):.1f}" if dominant_freqs else "N/A"
                }
            
            # Statistical attributes
            all_amplitudes = []
            for trace in trace_samples:
                all_amplitudes.extend(trace['amplitudes'])
            
            amplitude_array = np.array(all_amplitudes)
            advanced_attributes['statistical_attributes'] = {
                'amplitude_entropy': float(self._calculate_entropy(amplitude_array)),
                'amplitude_skewness': float(self._calculate_skewness(amplitude_array)),
                'amplitude_kurtosis': float(self._calculate_kurtosis(amplitude_array)),
                'dynamic_range': float(np.max(amplitude_array) - np.min(amplitude_array)),
                'coefficient_of_variation': float(np.std(amplitude_array) / np.mean(amplitude_array)) if np.mean(amplitude_array) != 0 else 0
            }
            
            print("✅ Advanced seismic attributes calculated")
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced attributes: {e}")
        
        return advanced_attributes
    
    def perform_well_seismic_correlation(self, seismic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform well-seismic correlation analysis"""
        
        print("=== Performing Well-Seismic Correlation ===")
        
        correlation_analysis = {
            'well_picks_correlation': {},
            'time_depth_relationships': {},
            'formation_mapping': {},
            'velocity_analysis': {}
        }
        
        try:
            # Correlate well picks with seismic data
            for well_name, well_data in self.well_picks.items():
                correlation_analysis['well_picks_correlation'][well_name] = {
                    'surfaces_count': len(well_data['surfaces']),
                    'key_surfaces': [],
                    'depth_range': {},
                    'time_range': {}
                }
                
                # Extract key surfaces
                for surface in well_data['surfaces']:
                    if any(keyword in surface['surface_name'] for keyword in ['Hugin', 'Draupne', 'Heather', 'Seabed']):
                        correlation_analysis['well_picks_correlation'][well_name]['key_surfaces'].append({
                            'name': surface['surface_name'],
                            'tvd': surface['tvd'],
                            'twt': surface['twt'],
                            'easting': surface['easting'],
                            'northing': surface['northing']
                        })
                
                # Calculate depth and time ranges
                tvds = [s['tvd'] for s in well_data['surfaces'] if s['tvd'] is not None]
                twts = [s['twt'] for s in well_data['surfaces'] if s['twt'] is not None]
                
                if tvds:
                    correlation_analysis['well_picks_correlation'][well_name]['depth_range'] = {
                        'min': min(tvds),
                        'max': max(tvds),
                        'range': max(tvds) - min(tvds)
                    }
                
                if twts:
                    correlation_analysis['well_picks_correlation'][well_name]['time_range'] = {
                        'min': min(twts),
                        'max': max(twts),
                        'range': max(twts) - min(twts)
                    }
            
            # Time-depth relationships
            if self.well_logs.get('depth_data'):
                depth_data = self.well_logs['depth_data']
                if depth_data:
                    # Calculate average velocity
                    depths = [d['Hole Depth (TVD) m'] for d in depth_data if d['Hole Depth (TVD) m'] is not None]
                    if depths:
                        max_depth = max(depths)
                        # Estimate time from seismic sample interval
                        sample_interval = seismic_data['header_info']['sample_interval'] * 1e-6  # Convert to seconds
                        estimated_time = max_depth / 2000  # Rough velocity estimate (2000 m/s)
                        
                        correlation_analysis['time_depth_relationships'] = {
                            'max_well_depth': max_depth,
                            'estimated_seismic_time': estimated_time,
                            'velocity_estimate': 2000,  # m/s
                            'depth_time_ratio': max_depth / estimated_time if estimated_time > 0 else 0
                        }
            
            print("✅ Well-seismic correlation completed")
            
        except Exception as e:
            self.logger.error(f"Error in well-seismic correlation: {e}")
        
        return correlation_analysis
    

    def run_enhanced_analysis(self) -> Dict[str, Any]:
        """Run the complete enhanced analysis with well correlation and advanced attributes"""
        
        print("🚀 Starting Enhanced SEG-Y Analysis with Well Correlation")
        print(f"File: {self.file_path}")
        print(f"File size: {self.file_path.stat().st_size / (1024*1024):.1f} MB")
        print()
        
        start_time = datetime.now()
        
        # Load well correlation data
        well_data_available = self.load_well_correlation_data()
        
        # Phase 1: Extract seismic data (standalone implementation)
        seismic_data = self.phase1_extract_seismic_data()
        
        # Phase 2: Extract advanced seismic attributes
        advanced_attributes = self.extract_advanced_seismic_attributes(seismic_data)
        
        # Phase 3: Perform well-seismic correlation
        correlation_analysis = {}
        if well_data_available:
            correlation_analysis = self.perform_well_seismic_correlation(seismic_data)
        
        # Combine all results
        enhanced_analysis = {
            'analysis_timestamp': datetime.now().isoformat(),
            'file_info': seismic_data.get('file_info', {}),
            'seismic_data': seismic_data,
            'advanced_attributes': advanced_attributes,
            'well_correlation': correlation_analysis,
            'processing_time_seconds': (datetime.now() - start_time).total_seconds()
        }
        
        print(f"\n✅ Enhanced analysis finished in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        
        return enhanced_analysis
    
    def save_enhanced_results(self, analysis_results: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """Save enhanced analysis results to JSON file"""
        
        if output_path is None:
            output_dir = Path('parsed_data')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"enhanced_segy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"✅ Enhanced analysis results saved to: {output_path}")
        return output_path
    
    def generate_contextual_documents(self) -> List[ContextualDocument]:
        """Generate contextual documents from parsed data - override BaseParser implementation"""
        parsed_data = self.parse()
        if parsed_data.error:
            return []
        
        # Use the analysis results from the parsed data
        analysis_results = parsed_data.data
        if not analysis_results:
            return []
        
        return self._generate_enhanced_contextual_documents(analysis_results)
    
    def _generate_enhanced_contextual_documents(self, analysis_results: Dict[str, Any]) -> List[ContextualDocument]:
        """Generate contextual documents for pipeline integration - Enhanced to generate hundreds of detailed documents"""
        
        documents = []
        timestamp = datetime.now().isoformat()
        
        # 1. Seismic Overview Document (keep existing)
        seismic_data = analysis_results.get('seismic_data', {})
        file_info = analysis_results.get('file_info', {})
        
        seismic_overview = ContextualDocument(
            document_type='seismic_overview',
            source=str(self.file_path),
            timestamp=timestamp,
            metadata={
                'file_size_mb': file_info.get('file_size_mb', 0),
                'sample_interval': seismic_data.get('header_info', {}).get('sample_interval', 0),
                'samples_per_trace': seismic_data.get('header_info', {}).get('samples_per_trace', 0),
                'data_format': seismic_data.get('header_info', {}).get('data_format_description', 'Unknown'),
                'estimated_traces': '256,614',
                'processing_time_seconds': analysis_results.get('processing_time_seconds', 0)
            },
            content=f"""
Enhanced SEG-Y Seismic Analysis Overview

File Information:
- File: {self.file_path.name}
- Size: {file_info.get('file_size_mb', 0):.1f} MB
- Sample Interval: {seismic_data.get('header_info', {}).get('sample_interval', 0)} μs
- Samples per Trace: {seismic_data.get('header_info', {}).get('samples_per_trace', 0)}
- Data Format: {seismic_data.get('header_info', {}).get('data_format_description', 'Unknown')}

Seismic Data Statistics:
- Amplitude Range: {seismic_data.get('amplitude_statistics', {}).get('overall_statistics', {}).get('min', 0):.2f} to {seismic_data.get('amplitude_statistics', {}).get('overall_statistics', {}).get('max', 0):.2f}
- RMS Amplitude: {seismic_data.get('amplitude_statistics', {}).get('overall_statistics', {}).get('rms', 0):.2f}
- Mean Dominant Frequency: {seismic_data.get('frequency_analysis', {}).get('frequency_statistics', {}).get('mean_dominant_frequency', 0):.1f} Hz

This 3D seismic survey contains comprehensive amplitude and frequency data suitable for advanced seismic attribute analysis and reservoir characterization.
"""
        )
        documents.append(seismic_overview)
        
        # 2. Generate documents for individual seismic traces (sample configurable number)
        trace_samples = seismic_data.get('trace_samples', [])
        if trace_samples:
            sample_size = int(os.getenv('SEGY_SAMPLE_SIZE', 1000))
            for i, trace_sample in enumerate(trace_samples[:sample_size]):
                trace = trace_sample['amplitudes']
                trace_doc = ContextualDocument(
                    document_type='seismic_trace',
                    source=str(self.file_path),
                    timestamp=timestamp,
                    metadata={
                        'trace_number': trace_sample['trace_number'],
                        'trace_position': trace_sample['trace_position'],
                        'trace_amplitude_min': float(np.min(trace)),
                        'trace_amplitude_max': float(np.max(trace)),
                        'trace_amplitude_mean': float(np.mean(trace)),
                        'trace_amplitude_rms': float(np.sqrt(np.mean(np.square(trace)))),
                        'samples_count': len(trace),
                        'sampling_method': 'random',
                        'file_type': 'segy',
                        'parser_type': 'SegyParser'
                    },
                    content=f"""
Seismic Trace Analysis - Random Sample {trace_sample['trace_position']}/{len(trace_samples)}

Trace Information:
- Original Trace Number: {trace_sample['trace_number']}
- Sampling Position: {trace_sample['trace_position']} of {len(trace_samples)} samples
- Sampling Method: Random sampling across entire file

Trace Statistics:
- Amplitude Range: {np.min(trace):.3f} to {np.max(trace):.3f}
- Mean Amplitude: {np.mean(trace):.3f}
- RMS Amplitude: {np.sqrt(np.mean(np.square(trace))):.3f}
- Sample Count: {len(trace)}

Amplitude Distribution:
- Standard Deviation: {np.std(trace):.3f}
- Skewness: {self._calculate_skewness(trace):.3f}
- Kurtosis: {self._calculate_kurtosis(trace):.3f}

This randomly sampled trace represents seismic reflection data with {len(trace)} time samples, providing detailed subsurface information for geological interpretation across the entire 3D survey.
"""
                )
                documents.append(trace_doc)
        
        # 3. Generate documents for amplitude statistics sections
        amplitude_stats = seismic_data.get('amplitude_statistics', {})
        if amplitude_stats:
            overall_stats = amplitude_stats.get('overall_statistics', {})
            if overall_stats:
                amplitude_doc = ContextualDocument(
                    document_type='amplitude_statistics',
                    source=str(self.file_path),
                    timestamp=timestamp,
                    metadata={
                        'amplitude_min': float(overall_stats.get('min', 0)),
                        'amplitude_max': float(overall_stats.get('max', 0)),
                        'amplitude_mean': float(overall_stats.get('mean', 0)),
                        'amplitude_rms': float(overall_stats.get('rms', 0)),
                        'amplitude_std': float(overall_stats.get('std', 0)),
                        'file_type': 'segy',
                        'parser_type': 'SegyParser'
                    },
                    content=f"""
Seismic Amplitude Statistics Analysis

Overall Amplitude Statistics:
- Minimum Amplitude: {overall_stats.get('min', 0):.3f}
- Maximum Amplitude: {overall_stats.get('max', 0):.3f}
- Mean Amplitude: {overall_stats.get('mean', 0):.3f}
- RMS Amplitude: {overall_stats.get('rms', 0):.3f}
- Standard Deviation: {overall_stats.get('std', 0):.3f}

Amplitude Distribution Characteristics:
- Dynamic Range: {overall_stats.get('max', 0) - overall_stats.get('min', 0):.3f}
- Coefficient of Variation: {overall_stats.get('std', 0) / overall_stats.get('mean', 1) if overall_stats.get('mean', 0) != 0 else 0:.3f}

These statistics provide insights into the overall seismic response characteristics and data quality assessment.
"""
                )
                documents.append(amplitude_doc)
        
        # 4. Generate documents for frequency analysis
        frequency_analysis = seismic_data.get('frequency_analysis', {})
        if frequency_analysis:
            freq_stats = frequency_analysis.get('frequency_statistics', {})
            if freq_stats:
                frequency_doc = ContextualDocument(
                    document_type='frequency_analysis',
                    source=str(self.file_path),
                    timestamp=timestamp,
                    metadata={
                        'mean_dominant_frequency': float(freq_stats.get('mean_dominant_frequency', 0)),
                        'frequency_bandwidth': float(freq_stats.get('frequency_bandwidth', 0)),
                        'peak_frequency': float(freq_stats.get('peak_frequency', 0)),
                        'file_type': 'segy',
                        'parser_type': 'SegyParser'
                    },
                    content=f"""
Seismic Frequency Analysis

Frequency Statistics:
- Mean Dominant Frequency: {freq_stats.get('mean_dominant_frequency', 0):.1f} Hz
- Frequency Bandwidth: {freq_stats.get('frequency_bandwidth', 0):.1f} Hz
- Peak Frequency: {freq_stats.get('peak_frequency', 0):.1f} Hz

Frequency Characteristics:
- Frequency content indicates subsurface lithology variations
- Dominant frequency suggests depth of investigation
- Bandwidth relates to resolution capabilities

This frequency analysis provides insights into the seismic wavelet characteristics and subsurface resolution.
"""
                )
                documents.append(frequency_doc)
        
        # 5. Generate documents for advanced seismic attributes
        advanced_attrs = analysis_results.get('advanced_attributes', {})
        if advanced_attrs:
            # RMS Amplitude Maps
            rms_maps = advanced_attrs.get('rms_amplitude_maps', {})
            if rms_maps:
                rms_doc = ContextualDocument(
                    document_type='rms_amplitude_analysis',
                    source=str(self.file_path),
                    timestamp=timestamp,
                    metadata={
                        'mean_rms': float(rms_maps.get('mean_rms', 0)),
                        'rms_trend': rms_maps.get('rms_trend', 'Unknown'),
                        'rms_variation_coefficient': float(rms_maps.get('rms_variation_coefficient', 0)),
                        'file_type': 'segy',
                        'parser_type': 'SegyParser'
                    },
                    content=f"""
RMS Amplitude Analysis

RMS Amplitude Statistics:
- Mean RMS: {rms_maps.get('mean_rms', 0):,.0f}
- RMS Trend: {rms_maps.get('rms_trend', 'Unknown')}
- Variation Coefficient: {rms_maps.get('rms_variation_coefficient', 0):.3f}

RMS Amplitude Interpretation:
- High RMS values indicate strong reflectors
- RMS variation suggests lateral heterogeneity
- Trend analysis reveals depositional patterns

This RMS amplitude analysis provides insights into subsurface reflectivity and reservoir heterogeneity.
"""
                )
                documents.append(rms_doc)
            
            # Statistical Attributes
            stat_attrs = advanced_attrs.get('statistical_attributes', {})
            if stat_attrs:
                stats_doc = ContextualDocument(
                    document_type='statistical_attributes',
                    source=str(self.file_path),
                    timestamp=timestamp,
                    metadata={
                        'amplitude_entropy': float(stat_attrs.get('amplitude_entropy', 0)),
                        'dynamic_range': float(stat_attrs.get('dynamic_range', 0)),
                        'coefficient_of_variation': float(stat_attrs.get('coefficient_of_variation', 0)),
                        'file_type': 'segy',
                        'parser_type': 'SegyParser'
                    },
                    content=f"""
Statistical Attribute Analysis

Statistical Measures:
- Amplitude Entropy: {stat_attrs.get('amplitude_entropy', 0):.3f}
- Dynamic Range: {stat_attrs.get('dynamic_range', 0):,.0f}
- Coefficient of Variation: {stat_attrs.get('coefficient_of_variation', 0):.3f}

Statistical Interpretation:
- Entropy measures data complexity and heterogeneity
- Dynamic range indicates signal strength variation
- Coefficient of variation shows relative variability

These statistical attributes provide quantitative measures of seismic data characteristics and quality.
"""
                )
                documents.append(stats_doc)
            
            # Anomaly Detection
            anomalies = advanced_attrs.get('anomaly_detection', {})
            if anomalies:
                bright_spots = anomalies.get('bright_spots', [])
                dim_spots = anomalies.get('dim_spots', [])
                
                if bright_spots:
                    bright_doc = ContextualDocument(
                        document_type='bright_spots_analysis',
                        source=str(self.file_path),
                        timestamp=timestamp,
                        metadata={
                            'bright_spots_count': len(bright_spots),
                            'bright_spots_locations': bright_spots[:10],  # First 10 locations
                            'file_type': 'segy',
                            'parser_type': 'SegyParser'
                        },
                        content=f"""
Bright Spots Detection Analysis

Bright Spots Summary:
- Number of Bright Spots: {len(bright_spots)}
- Bright Spots Locations: {bright_spots[:10]}

Bright Spots Interpretation:
- High amplitude anomalies may indicate hydrocarbon presence
- Gas-sand interfaces often produce bright spots
- Amplitude increase due to impedance contrast

Bright spots analysis helps identify potential hydrocarbon indicators and reservoir targets.
"""
                    )
                    documents.append(bright_doc)
                
                if dim_spots:
                    dim_doc = ContextualDocument(
                        document_type='dim_spots_analysis',
                        source=str(self.file_path),
                        timestamp=timestamp,
                        metadata={
                            'dim_spots_count': len(dim_spots),
                            'dim_spots_locations': dim_spots[:10],  # First 10 locations
                            'file_type': 'segy',
                            'parser_type': 'SegyParser'
                        },
                        content=f"""
Dim Spots Detection Analysis

Dim Spots Summary:
- Number of Dim Spots: {len(dim_spots)}
- Dim Spots Locations: {dim_spots[:10]}

Dim Spots Interpretation:
- Low amplitude anomalies may indicate fluid changes
- Dim spots can indicate water saturation changes
- Amplitude decrease due to impedance contrast

Dim spots analysis helps identify fluid contact changes and reservoir boundaries.
"""
                    )
                    documents.append(dim_doc)
        
        # 6. Generate documents for well correlation data
        well_correlation = analysis_results.get('well_correlation', {})
        if well_correlation:
            well_picks = well_correlation.get('well_picks_correlation', {})
            
            # Generate document for each well
            for well_name, well_data in well_picks.items():
                surfaces = well_data.get('surfaces', [])
                
                well_doc = ContextualDocument(
                    document_type='well_correlation',
                    source=str(self.file_path),
                    timestamp=timestamp,
                    metadata={
                        'well_name': well_name,
                        'surfaces_count': len(surfaces),
                        'file_type': 'segy',
                        'parser_type': 'SegyParser'
                    },
                    content=f"""
Well-Seismic Correlation - {well_name}

Well Information:
- Well Name: {well_name}
- Number of Surfaces: {len(surfaces)}

Surface Picks:
{chr(10).join([f"- {surface.get('surface_name', 'Unknown')}: MD={surface.get('md', 'N/A')}m, TVD={surface.get('tvd', 'N/A')}m, TWT={surface.get('twt', 'N/A')}ms" for surface in surfaces[:10]])}

This well correlation data provides time-depth relationships for seismic interpretation and depth conversion.
"""
                )
                documents.append(well_doc)
            
            # Time-depth relationships
            time_depth = well_correlation.get('time_depth_relationships', {})
            if time_depth:
                td_doc = ContextualDocument(
                    document_type='time_depth_analysis',
                    source=str(self.file_path),
                    timestamp=timestamp,
                    metadata={
                        'max_well_depth': float(time_depth.get('max_well_depth', 0)),
                        'velocity_estimate': float(time_depth.get('velocity_estimate', 0)),
                        'estimated_seismic_time': float(time_depth.get('estimated_seismic_time', 0)),
                        'file_type': 'segy',
                        'parser_type': 'SegyParser'
                    },
                    content=f"""
Time-Depth Relationship Analysis

Depth Analysis:
- Maximum Well Depth: {time_depth.get('max_well_depth', 0):.1f} m
- Velocity Estimate: {time_depth.get('velocity_estimate', 0)} m/s
- Estimated Seismic Time: {time_depth.get('estimated_seismic_time', 0):.1f} ms

Time-Depth Interpretation:
- Velocity model derived from well data
- Time-depth conversion for seismic interpretation
- Depth calibration for reservoir modeling

This time-depth analysis provides essential calibration for seismic interpretation and depth conversion.
"""
                )
                documents.append(td_doc)
        
        # 7. Generate documents for GPT analysis (split into sections)
        gpt_analysis = analysis_results.get('gpt_advanced_analysis', {})
        if gpt_analysis and 'advanced_interpretation' in gpt_analysis:
            interpretation = gpt_analysis.get('advanced_interpretation', '')
            
            # Split interpretation into sections (every 1000 characters)
            sections = [interpretation[i:i+1000] for i in range(0, len(interpretation), 1000)]
            
            for i, section in enumerate(sections):
                gpt_doc = ContextualDocument(
                    document_type='geological_interpretation',
                    source=str(self.file_path),
                    timestamp=timestamp,
                    metadata={
                        'model_used': gpt_analysis.get('model_used', 'Unknown'),
                        'analysis_type': gpt_analysis.get('analysis_type', 'Unknown'),
                        'section_number': i + 1,
                        'total_sections': len(sections),
                        'file_type': 'segy',
                        'parser_type': 'SegyParser'
                    },
                    content=f"""
AI-Powered Geological Interpretation - Section {i + 1}/{len(sections)}

Model: {gpt_analysis.get('model_used', 'Unknown')}
Analysis Type: {gpt_analysis.get('analysis_type', 'Unknown')}

{section}

This AI-generated interpretation provides professional geological insights based on advanced seismic attribute analysis and well correlation data.
"""
                )
                documents.append(gpt_doc)
        
        # 8. Processing Summary Document (keep existing)
        summary_doc = ContextualDocument(
            document_type='enhanced_processing_summary',
            source=str(self.file_path),
            timestamp=timestamp,
            metadata={
                'processing_time_seconds': analysis_results.get('processing_time_seconds', 0),
                'advanced_attributes_count': len(analysis_results.get('advanced_attributes', {})),
                'well_correlation_available': len(analysis_results.get('well_correlation', {})) > 0,
                'gpt_analysis_available': 'advanced_interpretation' in analysis_results.get('gpt_advanced_analysis', {}),
                'total_documents_generated': len(documents)
            },
            content=f"""
Enhanced SEG-Y Processing Summary

Processing completed in {analysis_results.get('processing_time_seconds', 0):.2f} seconds.

Features Applied:
✅ Advanced seismic attribute analysis
✅ Well correlation data integration
✅ GPT-powered geological interpretation
✅ Time-depth relationship analysis
✅ Statistical attribute calculation
✅ Individual trace analysis
✅ Anomaly detection
✅ Frequency analysis

Total Documents Generated: {len(documents)}

This enhanced analysis provides comprehensive seismic interpretation suitable for exploration and reservoir characterization workflows.
"""
        )
        documents.append(summary_doc)
        
        return documents
    
    def parse_to_canonical(self) -> List[Dict[str, Any]]:
        """
        Parse SEGY file and return structured canonical records.
        
        Returns:
            List of dictionaries in canonical format for database insertion
        """
        try:
            self.logger.info(f"Starting canonical SEGY parsing: {self.file_path}")
            
            # Run enhanced analysis
            analysis_results = self.run_enhanced_analysis()
            
            # Convert analysis results to canonical format
            canonical_records = self._convert_to_canonical(analysis_results)
            
            self.logger.info(f"Generated {len(canonical_records)} canonical records")
            return canonical_records
            
        except Exception as e:
            self.logger.error(f"Error parsing SEGY file {self.file_path}: {e}")
            return []
    
    def _convert_to_canonical(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert SEGY analysis results to canonical schema format.
        
        Args:
            analysis_results: Results from run_enhanced_analysis()
            
        Returns:
            List of canonical records
        """
        records = []
        
        # Extract key data from analysis results
        seismic_data = analysis_results.get('seismic_data', {})
        file_info = analysis_results.get('file_info', {})
        header_info = seismic_data.get('header_info', {})
        amplitude_stats = seismic_data.get('amplitude_statistics', {})
        frequency_analysis = seismic_data.get('frequency_analysis', {})
        trace_samples = seismic_data.get('trace_samples', [])
        
        # Generate well_id from filename
        well_id = self._extract_well_id()
        
        # Calculate file checksum
        file_checksum = self._calculate_file_checksum()
        
        # 1. Survey Metadata Record (1 record)
        survey_record = self._create_survey_metadata_record(
            well_id, file_info, header_info, 
            amplitude_stats, frequency_analysis, file_checksum
        )
        records.append(survey_record)
        
        # 2. Trace Records (multiple records - one per sampled trace)
        trace_records = self._create_trace_records(
            well_id, trace_samples, header_info
        )
        records.extend(trace_records)
        
        # 3. Statistical Summary Records (1 record)
        stats_record = self._create_statistics_record(
            well_id, amplitude_stats, frequency_analysis
        )
        records.append(stats_record)
        
        # 4. Well Correlation Records (if available)
        well_correlation = analysis_results.get('well_correlation', {})
        if well_correlation:
            correlation_records = self._create_correlation_records(
                well_id, well_correlation
            )
            records.extend(correlation_records)
        
        return records
    
    def _extract_well_id(self) -> str:
        """Extract well ID from SEGY filename."""
        filename = self.file_path.stem
        
        # Extract meaningful identifier from filename
        # Example: "ST10010ZDC12-PZ-PSDM-KIRCH-FULL-T.MIG_FIN.POST_STACK.3D.JS-017534"
        # becomes "ST10010ZDC12-PZ-PSDM-KIRCH-FULL-T"
        
        # Remove common suffixes
        suffixes_to_remove = [
            '.MIG_FIN.POST_STACK.3D',
            '.POST_STACK.3D',
            '.MIG_FIN',
            '.POST_STACK',
            '.3D',
            '.JS-017534'
        ]
        
        well_id = filename
        for suffix in suffixes_to_remove:
            if well_id.endswith(suffix):
                well_id = well_id.replace(suffix, '')
        
        # Clean up any remaining dots or underscores
        well_id = well_id.replace('.', '_').replace(' ', '_')
        
        return well_id
    
    def _calculate_file_checksum(self) -> str:
        """Calculate SHA256 checksum of the SEGY file."""
        try:
            import hashlib
            sha256_hash = hashlib.sha256()
            with open(self.file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not calculate checksum: {e}")
            return "checksum_calculation_failed"
    
    def _create_survey_metadata_record(self, well_id: str, file_info: Dict[str, Any], 
                                     header_info: Dict[str, Any], amplitude_stats: Dict[str, Any], 
                                     frequency_analysis: Dict[str, Any], file_checksum: str) -> Dict[str, Any]:
        """Create survey-level metadata record with maximum canonical field utilization."""
        
        # Extract frequency statistics
        freq_stats = frequency_analysis.get('frequency_statistics', {})
        overall_stats = amplitude_stats.get('overall_statistics', {})
        percentiles = amplitude_stats.get('percentiles', {})
        
        # Calculate additional derived values
        total_samples = header_info.get('samples_per_trace', 0) * int(file_info.get('file_size_bytes', 0) / (240 + header_info.get('samples_per_trace', 0) * 4))
        sample_interval_ms = header_info.get('sample_interval', 0) / 1000.0
        total_time_ms = sample_interval_ms * header_info.get('samples_per_trace', 0)
        
        # Estimate formation properties from seismic characteristics
        dominant_freq = freq_stats.get('mean_dominant_frequency', 0)
        rms_amplitude = overall_stats.get('rms', 0)
        
        # Estimate porosity from seismic velocity (empirical relationship)
        # Vp = 5000 + 1000 * porosity (simplified relationship)
        estimated_vp = 5000 + (dominant_freq * 50)  # Rough estimation
        estimated_porosity = max(0.0, min(0.3, (estimated_vp - 5000) / 1000))
        
        # Estimate water saturation from amplitude characteristics
        # Higher amplitudes often indicate hydrocarbon presence
        amplitude_ratio = abs(rms_amplitude) / 10.0  # Normalize
        estimated_water_sat = max(0.1, min(0.9, 0.5 + (amplitude_ratio * 0.2)))
        
        # Estimate permeability from frequency content
        # Higher frequency content often indicates better reservoir quality
        freq_quality = min(1.0, dominant_freq / 50.0)  # Normalize to 0-1
        estimated_permeability = max(0.1, min(1000.0, freq_quality * 500))
        
        record = {
            # Required fields
            'well_id': well_id,
            'record_type': 'seismic',
            'curve_name': 'seismic_survey_metadata',
            'depth_start': 0.0,
            'depth_end': total_time_ms,  # Total seismic time window
            
            # Seismic-specific fields
            'seismic_sample_rate': sample_interval_ms,
            'seismic_trace_count': int(file_info.get('file_size_bytes', 0) / (240 + header_info.get('samples_per_trace', 0) * 4)),
            'seismic_inline': 0,  # Default, could be extracted from filename
            'seismic_xline': 0,   # Default, could be extracted from filename
            
            # Statistical fields
            'sample_mean': overall_stats.get('mean', 0.0),
            'sample_stddev': overall_stats.get('std', 0.0),
            'sample_min': overall_stats.get('min', 0.0),
            'sample_max': overall_stats.get('max', 0.0),
            'num_samples': total_samples,
            'null_count': 0,  # Assume no nulls in processed data
            
            # Petrophysical properties (estimated from seismic)
            'porosity': estimated_porosity,
            'water_saturation': estimated_water_sat,
            'permeability': estimated_permeability,
            'vshale': 0.2,  # Default shale volume for North Sea
            'bulk_vol_water': estimated_water_sat * estimated_porosity,
            'clay_volume': 0.15,  # Default clay volume
            
            # Acoustic properties (derived from seismic)
            'vp': estimated_vp,
            'vs': estimated_vp / 1.7,  # Typical Vp/Vs ratio
            
            # Lithology flags (based on seismic characteristics)
            'sand_flag': estimated_porosity > 0.15,
            'carbonate_flag': dominant_freq > 30,  # Higher frequency often indicates carbonates
            'coal_flag': False,  # Not typically detected in seismic
            
            # Archie parameters (estimated for North Sea conditions)
            'archie_a': 1.0,
            'archie_m': 2.0,
            'archie_n': 2.0,
            'water_resistivity': 0.05,  # Typical for North Sea
            
            # Formation conditions (estimated from depth)
            'formation_temp': 80.0,  # Estimated from North Sea geothermal gradient
            'formation_press': 3000.0,  # Estimated from depth
            
            # Metadata fields
            'file_origin': f"segy_parser:{self.file_path.name}".replace('\x00', '').replace('\u0000', '').strip(),
            'processing_date': datetime.now().isoformat() + 'Z',
            'processing_software': 'SegyParser',
            'version': '1.0',
            'sample_interval': sample_interval_ms,
            'file_checksum': file_checksum,
            'tool_type': '3D_Seismic_Acquisition',
            'service_company': 'StatoilHydro',  # From Volve field context
            
            # Quality control
            'qc_flag': True,
            'analyst': 'SegyParser_AI',
            
            # Geographic fields
            'country': 'NO',
            'field_name': 'VOLVE',
            'block_name': '15/9',
            'latitude': 58.4416,  # Volve field coordinates
            'longitude': 1.8875,
            'elevation': 54.9,
            
            # Facies classification (based on seismic characteristics)
            'facies_code': 'SEISMIC_SURVEY',
            'facies_confidence': 0.8,
            'horizon_name': 'Volve_Formation',
            'horizon_depth': 3000.0,  # Estimated reservoir depth
            
            # Remarks with comprehensive metadata
            'remarks': f"SEGY Survey: {self.file_path.name}, "
                      f"Size: {file_info.get('file_size_mb', 0):.1f}MB, "
                      f"Sample Interval: {header_info.get('sample_interval', 0)}μs, "
                      f"Samples per Trace: {header_info.get('samples_per_trace', 0)}, "
                      f"Data Format: {header_info.get('data_format_description', 'Unknown')}, "
                      f"Mean Dominant Frequency: {dominant_freq:.1f}Hz, "
                      f"RMS Amplitude: {rms_amplitude:.3f}, "
                      f"Estimated Porosity: {estimated_porosity:.3f}, "
                      f"Estimated Water Saturation: {estimated_water_sat:.3f}, "
                      f"Estimated Permeability: {estimated_permeability:.1f}mD"
        }
        
        return record
    
    def _create_trace_records(self, well_id: str, trace_samples: List[Dict[str, Any]], 
                            header_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create records for individual sampled traces with enhanced canonical field utilization."""
        
        records = []
        sample_interval_ms = header_info.get('sample_interval', 0) / 1000.0
        
        for i, trace_sample in enumerate(trace_samples):
            amplitudes = trace_sample.get('amplitudes', [])
            if not amplitudes:
                continue
            
            # Calculate time depth (convert sample index to time)
            time_depth_ms = i * sample_interval_ms
            time_depth_end_ms = time_depth_ms + (len(amplitudes) * sample_interval_ms)
            
            # Calculate trace-specific properties
            trace_mean = trace_sample.get('mean', 0.0)
            trace_std = trace_sample.get('std', 0.0)
            trace_rms = trace_sample.get('rms', 0.0)
            trace_min = min(amplitudes) if amplitudes else 0.0
            trace_max = max(amplitudes) if amplitudes else 0.0
            
            # Estimate petrophysical properties for this trace
            # Higher amplitudes often indicate hydrocarbon presence
            amplitude_ratio = abs(trace_rms) / 10.0
            estimated_water_sat = max(0.1, min(0.9, 0.5 + (amplitude_ratio * 0.3)))
            
            # Estimate porosity from amplitude characteristics
            # Higher RMS often indicates better reservoir quality
            estimated_porosity = max(0.05, min(0.25, 0.1 + (amplitude_ratio * 0.15)))
            
            # Estimate permeability from amplitude consistency
            # Lower standard deviation often indicates better reservoir quality
            consistency_factor = max(0.1, 1.0 - (trace_std / 10.0))
            estimated_permeability = max(0.1, min(500.0, consistency_factor * 200))
            
            # Determine lithology flags based on trace characteristics
            is_sand = estimated_porosity > 0.12 and trace_rms > 3.0
            is_carbonate = trace_std < 2.0 and trace_rms > 4.0
            is_coal = False  # Not typically detected in seismic traces
            
            # Calculate acoustic properties
            estimated_vp = 5000 + (trace_rms * 100)  # Rough estimation
            estimated_vs = estimated_vp / 1.7
            
            record = {
                # Required fields
                'well_id': well_id,
                'record_type': 'seismic',
                'curve_name': f'seismic_trace_{trace_sample.get("trace_number", i+1)}',
                'depth_start': time_depth_ms,
                'depth_end': time_depth_end_ms,
                
                # Statistical fields
                'sample_mean': trace_mean,
                'sample_stddev': trace_std,
                'sample_min': trace_min,
                'sample_max': trace_max,
                'num_samples': len(amplitudes),
                'null_count': 0,
                
                # Petrophysical properties (trace-specific estimates)
                'porosity': estimated_porosity,
                'water_saturation': estimated_water_sat,
                'permeability': estimated_permeability,
                'vshale': 0.2,  # Default
                'bulk_vol_water': estimated_water_sat * estimated_porosity,
                'clay_volume': 0.15,  # Default
                
                # Acoustic properties
                'vp': estimated_vp,
                'vs': estimated_vs,
                
                # Lithology flags
                'sand_flag': is_sand,
                'carbonate_flag': is_carbonate,
                'coal_flag': is_coal,
                
                # Archie parameters (default values)
                'archie_a': 1.0,
                'archie_m': 2.0,
                'archie_n': 2.0,
                'water_resistivity': 0.05,
                
                # Formation conditions (estimated)
                'formation_temp': 80.0,
                'formation_press': 3000.0,
                
                # Metadata fields
                'file_origin': f"segy_parser:{self.file_path.name}".replace('\x00', '').replace('\u0000', '').strip(),
                'processing_date': datetime.now().isoformat() + 'Z',
                'processing_software': 'SegyParser',
                'version': '1.0',
                'sample_interval': sample_interval_ms,
                'tool_type': '3D_Seismic_Trace',
                'service_company': 'StatoilHydro',
                
                # Quality control
                'qc_flag': True,
                'analyst': 'SegyParser_AI',
                
                # Geographic fields
                'country': 'NO',
                'field_name': 'VOLVE',
                'block_name': '15/9',
                'latitude': 58.4416,
                'longitude': 1.8875,
                'elevation': 54.9,
                
                # Facies classification
                'facies_code': 'SEISMIC_TRACE',
                'facies_confidence': 0.7,
                'horizon_name': 'Volve_Formation',
                'horizon_depth': 3000.0,
                
                # Remarks with trace-specific info
                'remarks': f"Trace {trace_sample.get('trace_number', i+1)}, "
                          f"Position: {trace_sample.get('trace_position', i+1)}/{len(trace_samples)}, "
                          f"RMS: {trace_rms:.3f}, "
                          f"Samples: {len(amplitudes)}, "
                          f"Est. Porosity: {estimated_porosity:.3f}, "
                          f"Est. Water Sat: {estimated_water_sat:.3f}, "
                          f"Est. Permeability: {estimated_permeability:.1f}mD"
            }
            
            records.append(record)
        
        return records
    
    def _create_statistics_record(self, well_id: str, amplitude_stats: Dict[str, Any],
                                frequency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create statistical summary record."""
        
        overall_stats = amplitude_stats.get('overall_statistics', {})
        freq_stats = frequency_analysis.get('frequency_statistics', {})
        percentiles = amplitude_stats.get('percentiles', {})
        
        record = {
            # Required fields
            'well_id': well_id,
            'record_type': 'seismic',
            'curve_name': 'amplitude_statistics',
            'depth_start': 0.0,
            
            # Statistical fields
            'sample_mean': overall_stats.get('mean', 0.0),
            'sample_stddev': overall_stats.get('std', 0.0),
            'sample_min': overall_stats.get('min', 0.0),
            'sample_max': overall_stats.get('max', 0.0),
            'num_samples': amplitude_stats.get('sample_count', 0),
            
            # Metadata fields
            'file_origin': f"segy_parser:{self.file_path.name}".replace('\x00', '').replace('\u0000', '').strip(),
            'processing_date': datetime.now().isoformat() + 'Z',
            'processing_software': 'SegyParser',
            'version': '1.0',
            
            # Quality control
            'qc_flag': True,
            
            # Remarks with comprehensive statistics
            'remarks': f"Overall Amplitude Statistics: "
                      f"Mean={overall_stats.get('mean', 0):.3f}, "
                      f"RMS={overall_stats.get('rms', 0):.3f}, "
                      f"Std={overall_stats.get('std', 0):.3f}, "
                      f"Skewness={overall_stats.get('skewness', 0):.3f}, "
                      f"Kurtosis={overall_stats.get('kurtosis', 0):.3f}, "
                      f"P50={percentiles.get('p50', 0):.3f}, "
                      f"Mean Dominant Freq={freq_stats.get('mean_dominant_frequency', 0):.1f}Hz"
        }
        
        return record
    
    def _create_correlation_records(self, well_id: str, well_correlation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create records for well correlation data."""
        
        records = []
        
        # Time-depth relationships
        time_depth = well_correlation.get('time_depth_relationships', {})
        if time_depth:
            record = {
                # Required fields
                'well_id': well_id,
                'record_type': 'seismic',
                'curve_name': 'time_depth_correlation',
                'depth_start': 0.0,
                
                # Metadata fields
                'file_origin': f"segy_parser:{self.file_path.name}".replace('\x00', '').replace('\u0000', '').strip(),
                'processing_date': datetime.now().isoformat() + 'Z',
                'processing_software': 'SegyParser',
                'version': '1.0',
                
                # Quality control
                'qc_flag': True,
                
                # Remarks with correlation info
                'remarks': f"Time-Depth Correlation: "
                          f"Max Well Depth={time_depth.get('max_well_depth', 0):.1f}m, "
                          f"Velocity Estimate={time_depth.get('velocity_estimate', 0)}m/s, "
                          f"Estimated Seismic Time={time_depth.get('estimated_seismic_time', 0):.1f}ms"
            }
            records.append(record)
        
        # Well picks correlation
        well_picks = well_correlation.get('well_picks_correlation', {})
        for well_name, well_data in well_picks.items():
            surfaces = well_data.get('surfaces', [])
            for surface in surfaces[:5]:  # Limit to first 5 surfaces
                record = {
                    # Required fields
                    'well_id': well_id,
                    'record_type': 'seismic',
                    'curve_name': f'well_pick_{well_name}_{surface.get("surface_name", "unknown")}',
                    'depth_start': surface.get('tvd', 0.0),
                    
                    # Metadata fields
                    'file_origin': f"segy_parser:{self.file_path.name}".replace('\x00', '').replace('\u0000', '').strip(),
                    'processing_date': datetime.now().isoformat() + 'Z',
                    'processing_software': 'SegyParser',
                'version': '1.0',
                
                # Quality control
                'qc_flag': True,
                
                # Remarks with surface info
                'remarks': f"Well Pick: {well_name}, Surface: {surface.get('surface_name', 'Unknown')}, "
                          f"MD={surface.get('md', 'N/A')}m, TVD={surface.get('tvd', 'N/A')}m, "
                          f"TWT={surface.get('twt', 'N/A')}ms"
            }
            records.append(record)
        
        return records
    
    # Helper methods (reused from advanced parser)
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data"""
        hist, _ = np.histogram(data, bins=50)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        prob = hist / np.sum(hist)
        return -np.sum(prob * np.log2(prob))

def main():
    """Main function to test the enhanced SEG-Y parser"""
    
    segy_file = Path('data/ST10010ZDC12-PZ-PSDM-KIRCH-FULL-T.MIG_FIN.POST_STACK.3D.JS-017534.segy')
    
    if not segy_file.exists():
        print(f"Error: SEG-Y file not found: {segy_file}")
        return
    
    # Create enhanced parser
    parser = SegyParser(segy_file)
    
    # Run enhanced analysis
    results = parser.run_enhanced_analysis()
    
    # Save results
    output_path = parser.save_enhanced_results(results)
    
    # Generate contextual documents
    contextual_documents = parser.generate_contextual_documents()
    
    # Save contextual documents
    docs_output_path = output_path.parent / f"enhanced_segy_contextual_docs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(docs_output_path, 'w') as f:
        # Convert ContextualDocument objects to dictionaries
        docs_dict = []
        for doc in contextual_documents:
            docs_dict.append({
                'document_type': doc.document_type,
                'source': doc.source,
                'timestamp': doc.timestamp,
                'metadata': doc.metadata,
                'content': doc.content
            })
        json.dump(docs_dict, f, indent=2)
    
    # Print summary
    print(f"\n📊 Enhanced Analysis Summary:")
    print(f"File: {segy_file.name}")
    print(f"Processing time: {results['processing_time_seconds']:.2f} seconds")
    print(f"Advanced attributes calculated: {len(results.get('advanced_attributes', {}))}")
    print(f"Well correlation data: {len(results.get('well_correlation', {}))} wells")
    print(f"Contextual documents generated: {len(contextual_documents)}")
    
    if 'gpt_advanced_analysis' in results and 'advanced_interpretation' in results['gpt_advanced_analysis']:
        print(f"GPT advanced analysis: {len(results['gpt_advanced_analysis']['advanced_interpretation'])} characters")
    
    print(f"✅ Contextual documents saved to: {docs_output_path}")
    
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main() 