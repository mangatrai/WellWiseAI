# parsers/dlis.py

import os
import hashlib
import numpy as np
from datetime import datetime
from dlisio.dlis import load
import logging

from schema.fields import CANONICAL_FIELDS

def compute_checksum(path: str) -> str:
    """Compute SHA1 checksum of file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_get_attr(obj, attr_name, default=""):
    """Safely get attribute value with proper error handling."""
    try:
        value = getattr(obj, attr_name, default)
        return value
    except Exception:
        return default

def dms_to_decimal(dms_str):
    """Convert DMS (Degrees, Minutes, Seconds) format to decimal degrees."""
    try:
        # Handle format like "058 26' 29.706" N    DMS"
        if not dms_str or 'DMS' not in dms_str:
            return None
            
        # Extract the DMS part
        dms_part = dms_str.split('DMS')[0].strip()
        
        # Parse degrees, minutes, seconds
        parts = dms_part.replace('"', '').replace("'", ' ').split()
        
        if len(parts) >= 4:  # degrees, minutes, seconds, direction
            degrees = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            direction = parts[3]
            
            # Calculate decimal degrees
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            
            # Apply direction
            if direction in ['S', 'W']:
                decimal = -decimal
                
            return decimal
        else:
            return None
    except Exception:
        return None

def extract_channel_values(channel, logger):
    """Extract values from channel using dlisio API."""
    channel_name = safe_get_attr(channel, 'name', 'UNKNOWN')
    logger.debug(f"Extracting values for channel '{channel_name}'")
    
    try:
        # Get the curves data from the channel
        curves = channel.curves()
        
        if curves is not None and len(curves) > 0:
            # Convert to list and filter out null values
            values = []
            for val in curves:
                # Skip null values (typically -999.25 or similar)
                if val is not None and not np.isnan(val) and val > -999.0:
                    values.append(float(val))
            
            logger.debug(f"Extracted {len(values)} valid values from '{channel_name}'")
            return values
        else:
            logger.debug(f"Channel '{channel_name}' has no valid curves data")
            return []
            
    except Exception as e:
        logger.debug(f"Failed to extract values from '{channel_name}': {e}")
        return []

def extract_file_metadata(logical_file, logger):
    """Extract comprehensive file-level metadata from logical file."""
    file_metadata = {
        'job_id': "",
        'client_name': "",
        'formation': "",
        'log_date': "",
        'run_start_date': "",
        'run_stop_date': "",
        'software': "",
        'comments': "",
        'remarks': []  # For metadata that doesn't fit canonical fields
    }
    
    # Extract rich parameter data from logical file
    try:
        parameters = logical_file.parameters
        if parameters:
            logger.debug(f"Found {len(parameters)} parameters for metadata extraction")
            for param in parameters:
                param_name = safe_get_attr(param, 'name', '').upper()
                param_values = safe_get_attr(param, 'values', [])
                param_long_name = safe_get_attr(param, 'long_name', '')
                
                # Get the first value from the list
                param_value = param_values[0] if param_values else ""
                
                # Map to file metadata fields
                if param_name == 'CN' and not file_metadata['client_name']:
                    file_metadata['client_name'] = param_value
                elif param_name == 'WN' and not file_metadata['job_id']:
                    file_metadata['job_id'] = param_value
                elif param_name == 'FN' and not file_metadata['formation']:
                    file_metadata['formation'] = param_value
                elif param_name == 'RUN' and not file_metadata['log_date']:
                    file_metadata['log_date'] = param_value
                elif param_name == 'R1' and not file_metadata['software']:
                    file_metadata['software'] = param_value
                else:
                    # Add to remarks for non-canonical parameters
                    file_metadata['remarks'].append(f"Param_{param_name}: {param_value}")
                    
    except Exception as e:
        logger.debug(f"Error extracting parameter metadata: {e}")
    
    try:
        # Extract metadata from logical file - try different approaches
        metadata = None
        try:
            metadata = logical_file.metadata
        except AttributeError:
            # Try alternative access methods
            try:
                metadata = getattr(logical_file, 'metadata', None)
            except:
                pass
        
        if metadata:
            file_metadata['job_id'] = metadata.get("JOB_ID", "") or ""
            file_metadata['client_name'] = metadata.get("CLIENT", "") or ""
            file_metadata['formation'] = metadata.get("FORMATION", "") or ""
            file_metadata['log_date'] = metadata.get("LOGGED_DATE", "") or ""
            file_metadata['run_start_date'] = metadata.get("RUN_START_DATE", "") or ""
            file_metadata['run_stop_date'] = metadata.get("RUN_STOP_DATE", "") or ""
            file_metadata['software'] = metadata.get("SOFTWARE", "") or ""
        
        # Extract comments
        try:
            comments = logical_file.comments
            if comments:
                file_metadata['comments'] = "\n".join(comments)
        except Exception as e:
            logger.debug(f"Error extracting comments: {e}")
        
        # Extract additional metadata that doesn't fit canonical fields
        additional_metadata = []
        if metadata:
            for key, value in metadata.items():
                if key not in ["JOB_ID", "CLIENT", "FORMATION", "LOGGED_DATE", "RUN_START_DATE", "RUN_STOP_DATE", "SOFTWARE"]:
                    additional_metadata.append(f"{key}: {value}")
        
        if additional_metadata:
            file_metadata['remarks'].extend(additional_metadata)
            
    except Exception as e:
        logger.debug(f"Error extracting file metadata: {e}")
    
    return file_metadata

def extract_computations_and_parameters(logical_file, logger):
    """Extract computations and parameters from logical file."""
    comp_params = {
        'archie_a': None,
        'archie_m': None,
        'archie_n': None,
        'water_resistivity': None,
        'remarks': []  # For parameters that don't fit canonical fields
    }
    
    try:
        # Extract computations
        computations = logical_file.computations
        if computations:
            logger.debug(f"Found {len(computations)} computations")
            for comp in computations:
                mnemonic = safe_get_attr(comp, 'mnemonic', '').upper()
                params = safe_get_attr(comp, 'parameters', {})
                
                # Map to canonical fields
                if mnemonic == 'A' or 'ARCHIE_A' in mnemonic:
                    comp_params['archie_a'] = params.get('value', None)
                elif mnemonic == 'M' or 'ARCHIE_M' in mnemonic:
                    comp_params['archie_m'] = params.get('value', None)
                elif mnemonic == 'N' or 'ARCHIE_N' in mnemonic:
                    comp_params['archie_n'] = params.get('value', None)
                elif mnemonic == 'RW' or 'WATER_RESISTIVITY' in mnemonic:
                    comp_params['water_resistivity'] = params.get('value', None)
                else:
                    # Add to remarks for non-canonical parameters
                    comp_params['remarks'].append(f"Computation_{mnemonic}: {params}")
        
        # Extract parameters - FIXED: Use correct API structure
        parameters = logical_file.parameters
        if parameters:
            logger.debug(f"Found {len(parameters)} parameters")
            for param in parameters:
                # Use the correct attribute names from the debug output
                param_name = safe_get_attr(param, 'name', '').upper()
                param_values = safe_get_attr(param, 'values', [])
                param_long_name = safe_get_attr(param, 'long_name', '')
                
                # Skip empty names
                if not param_name:
                    continue
                
                # Get the first value from the list
                param_value = param_values[0] if param_values else ""
                
                # Map to canonical fields
                if param_name == 'A' or 'ARCHIE_A' in param_name:
                    comp_params['archie_a'] = param_value
                elif param_name == 'M' or 'ARCHIE_M' in param_name:
                    comp_params['archie_m'] = param_value
                elif param_name == 'N' or 'ARCHIE_N' in param_name:
                    comp_params['archie_n'] = param_value
                elif param_name == 'RW' or 'WATER_RESISTIVITY' in param_name:
                    comp_params['water_resistivity'] = param_value
                else:
                    # Add to remarks for non-canonical parameters
                    comp_params['remarks'].append(f"Parameter_{param_name}: {param_value}")
                    
    except Exception as e:
        logger.debug(f"Error extracting computations and parameters: {e}")
    
    return comp_params

def extract_calibrations_and_equipment(logical_file, logger):
    """Extract calibrations and equipment data from logical file."""
    calib_equip = {
        'calibration_scale': None,
        'calibration_offset': None,
        'equipment_model': "",
        'equipment_serial': "",
        'equipment_vendor': "",
        'remarks': []  # For data that doesn't fit canonical fields
    }
    
    try:
        # Extract calibrations
        calibrations = logical_file.calibrations
        if calibrations:
            logger.debug(f"Found {len(calibrations)} calibrations")
            for cal in calibrations:
                curve_name = safe_get_attr(cal, 'channel_name', '')
                scale = safe_get_attr(cal, 'scale_factor', None)
                offset = safe_get_attr(cal, 'offset', None)
                units = safe_get_attr(cal, 'units', '')
                
                # Store calibration data in remarks
                calib_equip['remarks'].append(f"Calibration_{curve_name}: scale={scale}, offset={offset}, units={units}")
        
        # Extract equipment
        equipments = logical_file.equipments
        if equipments:
            logger.debug(f"Found {len(equipments)} equipments")
            for eq in equipments:
                model = safe_get_attr(eq, 'model', '')
                serial = safe_get_attr(eq, 'serial_number', '')
                vendor = safe_get_attr(eq, 'manufacturer', '')
                
                # Store equipment data in remarks
                calib_equip['remarks'].append(f"Equipment: model={model}, serial={serial}, vendor={vendor}")
                
    except Exception as e:
        logger.debug(f"Error extracting calibrations and equipment: {e}")
    
    return calib_equip

def extract_zones(logical_file, logger):
    """Extract zone definitions from logical file."""
    zones_data = {
        'zone_name': "",
        'zone_top': None,
        'zone_base': None,
        'remarks': []  # For zone data that doesn't fit canonical fields
    }
    
    try:
        # Extract zones
        zones = logical_file.zones
        if zones:
            logger.debug(f"Found {len(zones)} zones")
            for zone in zones:
                name = safe_get_attr(zone, 'name', '')
                top = safe_get_attr(zone, 'top', None)
                base = safe_get_attr(zone, 'base', None)
                
                # Store zone data in remarks
                zones_data['remarks'].append(f"Zone_{name}: top={top}, base={base}")
                
    except Exception as e:
        logger.debug(f"Error extracting zones: {e}")
    
    return zones_data

def extract_origin_info(logical_file, logger):
    """Extract origin information from logical file."""
    origin_info = {
        'well_name': "",
        'company': "",
        'acquisition_date': "",
        'version': "",
        'service_company': "",
        'analyst': ""
    }
    
    try:
        # Get origins from the logical file
        origins = logical_file.origins
        
        if origins and len(origins) > 0:
            origin = origins[0]
            
            # Extract well information
            origin_info['well_name'] = safe_get_attr(origin, 'well_name', "") or ""
            origin_info['company'] = safe_get_attr(origin, 'company', "") or ""
            origin_info['service_company'] = safe_get_attr(origin, 'company', "") or ""
            origin_info['version'] = safe_get_attr(origin, 'version', "") or ""
            
            # Handle acquisition date
            ct = safe_get_attr(origin, 'creation_time', "")
            if ct and hasattr(ct, 'isoformat'):
                origin_info['acquisition_date'] = ct.isoformat()
            elif ct:
                origin_info['acquisition_date'] = str(ct)
                
            # Extract analyst information if available
            origin_info['analyst'] = safe_get_attr(origin, 'analyst', "") or ""
            
    except Exception as e:
        logger.warning(f"Error extracting origin info: {e}")
    
    return origin_info

def extract_enhanced_tool_info(channel, logger):
    """Extract enhanced tool information from channel including curve metadata."""
    tool_info = {
        'tool_type': "",
        'processing_software': "",
        'curve_units': "",
        'curve_description': "",
        'curve_type': "",
        'remarks': ""
    }
    
    try:
        # Get basic tool information from channel
        tool_info['tool_type'] = safe_get_attr(channel, 'type', "") or ""
        tool_info['processing_software'] = safe_get_attr(channel, 'software', "") or ""
        tool_info['remarks'] = safe_get_attr(channel, 'long_name', "") or ""
        
        # Extract curve-level metadata
        tool_info['curve_units'] = safe_get_attr(channel, 'units', "") or ""
        
        # Try alternative access for units
        if not tool_info['curve_units']:
            try:
                attributes = channel.attributes
                if attributes and 'UNIT' in attributes:
                    tool_info['curve_units'] = attributes['UNIT']
            except:
                pass
        
        # Get curve description
        tool_info['curve_description'] = safe_get_attr(channel, 'long_name', "") or ""
        
        # Try alternative access for description
        if not tool_info['curve_description']:
            try:
                attributes = channel.attributes
                if attributes and 'DESCR' in attributes:
                    tool_info['curve_description'] = attributes['DESCR']
            except:
                pass
        
        # Get curve type
        tool_info['curve_type'] = safe_get_attr(channel, 'type', "") or ""
        
        # Try alternative access for type
        if not tool_info['curve_type']:
            try:
                properties = channel.properties
                if properties and 'type' in properties:
                    tool_info['curve_type'] = properties['type']
            except:
                pass
        
    except Exception as e:
        logger.debug(f"Error extracting enhanced tool info: {e}")
    
    return tool_info

def extract_geographic_info(logical_file, logger):
    """Extract geographic information from logical file and parameters."""
    geo_info = {
        'country': "",
        'state_province': "",
        'field_name': "",
        'block_name': "",
        'latitude': None,
        'longitude': None,
        'elevation': None,
        'service_company': ""
    }
    
    try:
        # Get origins for geographic data
        origins = logical_file.origins
        
        if origins and len(origins) > 0:
            origin = origins[0]
            
            # Extract field name from origin
            geo_info['field_name'] = safe_get_attr(origin, 'field_name', "") or ""
            
            # Try to get geographic info from origin attributes
            geo_info['country'] = safe_get_attr(origin, 'country', "") or ""
            geo_info['state_province'] = safe_get_attr(origin, 'state', "") or ""
            geo_info['block_name'] = safe_get_attr(origin, 'block_name', "") or ""
            
            # Try to extract coordinates
            lat = safe_get_attr(origin, 'latitude', None)
            lon = safe_get_attr(origin, 'longitude', None)
            elev = safe_get_attr(origin, 'elevation', None)
            
            if lat is not None:
                geo_info['latitude'] = float(lat)
            if lon is not None:
                geo_info['longitude'] = float(lon)
            if elev is not None:
                geo_info['elevation'] = float(elev)
        
        # Try to get geographic info from parameters
        try:
            parameters = logical_file.parameters
            
            if parameters:
                for param in parameters:
                    param_name = safe_get_attr(param, 'name', '').upper()
                    param_values = safe_get_attr(param, 'values', [])
                    
                    # Get the first value from the list
                    param_value = param_values[0] if param_values else ""
                    
                    # Map geographic parameters
                    if param_name == 'CTRY' and not geo_info['country']:
                        geo_info['country'] = param_value
                    elif param_name == 'STAT' and not geo_info['state_province']:
                        geo_info['state_province'] = param_value
                    elif param_name == 'FLD' and not geo_info['field_name']:
                        geo_info['field_name'] = param_value
                    elif param_name == 'LATI':
                        geo_info['latitude'] = dms_to_decimal(param_value)
                    elif param_name == 'LONG':
                        geo_info['longitude'] = dms_to_decimal(param_value)
                    elif param_name == 'ELEV' or param_name == 'ELZ':
                        try:
                            geo_info['elevation'] = float(param_value)
                        except (ValueError, TypeError):
                            pass
                    elif param_name == 'COUN' and not geo_info['service_company']:
                        geo_info['service_company'] = param_value
                        
        except Exception as e:
            logger.debug(f"Error extracting geographic info from parameters: {e}")
            
    except Exception as e:
        logger.debug(f"Error extracting geographic info: {e}")
    
    return geo_info

def find_depth_channel(channels, logger):
    """Find the depth channel from a list of channels."""
    depth_keywords = ['DEPTH', 'MD', 'TVD', 'TVDSS', 'TVDKB']
    
    for channel in channels:
        channel_name = safe_get_attr(channel, 'name', '').upper()
        
        for keyword in depth_keywords:
            if keyword in channel_name:
                logger.debug(f"Found depth channel: {channel_name}")
                return channel
    
    return None

def compute_statistics(values):
    """Compute basic statistics for a list of values."""
    if not values:
        return {
            'mean': None,
            'min': None,
            'max': None,
            'count': 0,
            'stddev': None
        }
    
    try:
        values_array = np.array(values)
        return {
            'mean': float(np.mean(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'count': len(values_array),
            'stddev': float(np.std(values_array))
        }
    except Exception:
        return {
            'mean': None,
            'min': None,
            'max': None,
            'count': 0,
            'stddev': None
        }

def map_channel_to_canonical(channel_name, values, depth_values, origin_info, tool_info, 
                            geo_info, file_metadata, comp_params, calib_equip, zones_data, file_info, logger):
    """Map DLIS channel data to canonical schema with enhanced metadata."""
    rec = {field: "" for field in CANONICAL_FIELDS}
    
    # Set hard-coded defaults for fields with no extraction logic
    rec.update({
        "facies_code": "UNKNOWN",
        "horizon_name": "UNKNOWN",
        "gas_oil_ratio": 0.0,
        "production_rate": 0.0
    })
    
    # Basic well information
    rec.update({
        "well_id": origin_info['well_name'],
        "field_name": geo_info['field_name'],
        "country": geo_info['country'],
        "state_province": geo_info['state_province'],
        "service_company": geo_info['service_company'],
        "acquisition_date": origin_info['acquisition_date'],
        "record_type": "logging",
        "curve_name": channel_name,
        "file_origin": file_info['filename'],
        "depth_start": depth_values[0] if depth_values else None,
        "depth_end": depth_values[-1] if depth_values else None,
        "tool_type": "CHANNEL"
    })
    
    # Add non-canonical fields to remarks (DLIS doesn't extract these values, so they remain empty)
    if 'remarks' not in rec:
        rec['remarks'] = ""
    rec['remarks'] += "api_number: (not extracted from DLIS); "
    rec['remarks'] += "uwi: (not extracted from DLIS); "
    rec['remarks'] += "location: (not extracted from DLIS); "
    rec['remarks'] += "step_size: (not extracted from DLIS); "
    
    # Geographic data
    if geo_info['latitude'] is not None:
        rec['latitude'] = geo_info['latitude']
    else:
        rec['latitude'] = 0.0
    
    if geo_info['longitude'] is not None:
        rec['longitude'] = geo_info['longitude']
    else:
        rec['longitude'] = 0.0
    
    if geo_info['elevation'] is not None:
        rec['elevation'] = geo_info['elevation']
    
    # File metadata
    if file_metadata['job_id']:
        rec['remarks'] = f"Job ID: {file_metadata['job_id']}; "
    if file_metadata['client_name']:
        rec['remarks'] += f"Client: {file_metadata['client_name']}; "
    if file_metadata['formation']:
        rec['remarks'] += f"Formation: {file_metadata['formation']}; "
    if file_metadata['log_date']:
        rec['remarks'] += f"Log Date: {file_metadata['log_date']}; "
    if file_metadata['software']:
        rec['remarks'] += f"Software: {file_metadata['software']}; "
    if file_metadata['comments']:
        rec['remarks'] += f"Comments: {file_metadata['comments']}; "
    
    # Computations and parameters
    if comp_params['archie_a'] is not None:
        rec['archie_a'] = comp_params['archie_a']
    if comp_params['archie_m'] is not None:
        rec['archie_m'] = comp_params['archie_m']
    if comp_params['archie_n'] is not None:
        rec['archie_n'] = comp_params['archie_n']
    if comp_params['water_resistivity'] is not None:
        rec['water_resistivity'] = comp_params['water_resistivity']
    
    # Add computation remarks
    if comp_params['remarks']:
        rec['remarks'] += f"Computation Params: {'; '.join(comp_params['remarks'])}; "
    
    # Calibrations and equipment
    if calib_equip['remarks']:
        rec['remarks'] += f"Calibration/Equipment: {'; '.join(calib_equip['remarks'])}; "
    
    # Zones
    if zones_data['remarks']:
        rec['remarks'] += f"Zones: {'; '.join(zones_data['remarks'])}; "
    
    # Curve metadata
    if tool_info['curve_units']:
        rec['remarks'] += f"Curve Units: {tool_info['curve_units']}; "
    if tool_info['curve_description']:
        rec['remarks'] += f"Curve Description: {tool_info['curve_description']}; "
    if tool_info['curve_type']:
        rec['remarks'] += f"Curve Type: {tool_info['curve_type']}; "
    
    # Compute statistics
    stats = compute_statistics(values)
    
    # Map curve data based on curve name
    curve_upper = channel_name.upper()
    mean = stats['mean']
    
    # Petrophysical properties
    if 'BVW' in curve_upper:
        rec['bulk_vol_water'] = mean
    elif 'SW' in curve_upper:
        rec['water_saturation'] = mean
    elif 'PHIF' in curve_upper or 'PHIT' in curve_upper:
        rec['porosity'] = mean
    elif 'RT' in curve_upper or 'RESISTIVITY' in curve_upper:
        rec['resistivity_deep'] = mean
    elif 'RW' in curve_upper:
        rec['water_resistivity'] = mean
    elif 'RHOB' in curve_upper:
        rec['density'] = mean
    elif 'NPHI' in curve_upper:
        rec['neutron_porosity'] = mean
    elif 'GR' in curve_upper or 'GAMMA' in curve_upper:
                        # Add gamma_ray to remarks instead of separate field
                if 'remarks' not in rec:
                    rec['remarks'] = ""
                rec['remarks'] += f"gamma_ray: {mean}; "
    elif 'DT' in curve_upper or 'SONIC' in curve_upper:
        # Add to remarks since sonic_transit_time is not in canonical schema
        if 'remarks' not in rec:
            rec['remarks'] = ""
        rec['remarks'] += f"sonic_transit_time: {mean}; "
    elif 'CALI' in curve_upper:
        rec['caliper'] = mean
    elif 'PEF' in curve_upper:
        rec['photoelectric_factor'] = mean
    elif 'DTS' in curve_upper:
        rec['shear_sonic'] = mean
    elif 'VSH' in curve_upper:
        rec['vshale'] = mean
    elif 'SAND_FLAG' in curve_upper:
        rec['sample_mean'] = mean
        rec['sand_flag'] = mean > 0.0 if not np.isnan(mean) else False
    elif 'CARB_FLAG' in curve_upper:
        rec['sample_mean'] = mean
        rec['carbonate_flag'] = mean > 0.0 if not np.isnan(mean) else False
    elif 'COAL_FLAG' in curve_upper:
        rec['sample_mean'] = mean
        rec['coal_flag'] = mean > 0.0 if not np.isnan(mean) else False
    elif 'KLOGH' in curve_upper:
        rec['permeability'] = mean
    
    # Map curve statistics to canonical fields
    if stats['mean'] is not None:
        rec['sample_mean'] = stats['mean']
    if stats['min'] is not None:
        rec['sample_min'] = stats['min']
    if stats['max'] is not None:
        rec['sample_max'] = stats['max']
    if stats['count'] is not None:
        rec['num_samples'] = stats['count']
    if stats['stddev'] is not None:
        rec['sample_stddev'] = stats['stddev']
    
    return rec

def parse_dlis(file_path: str, logger: logging.Logger) -> list[dict]:
    """
    Comprehensive DLIS parser: extract all available data from DLIS files
    and map to the 75-field canonical schema with enhanced metadata extraction.
    """
    logger.info(f"Starting to parse DLIS file: {file_path}")
    
    file_info = {
        'filename': os.path.basename(file_path),
        'checksum': compute_checksum(file_path),
        'processing_date': datetime.utcnow().isoformat()
    }
    
    records = []
    
    try:
        with load(str(file_path)) as logical_files:
            logger.info(f"Found {len(logical_files)} logical files")
            
            for lf_idx, lf in enumerate(logical_files):
                logger.info(f"Processing logical file {lf_idx + 1}")
                
                # Extract comprehensive metadata
                origin_info = extract_origin_info(lf, logger)
                logger.debug(f"Origin info: {origin_info}")
                
                geo_info = extract_geographic_info(lf, logger)
                logger.debug(f"Geographic info: {geo_info}")
                
                file_metadata = extract_file_metadata(lf, logger)
                logger.debug(f"File metadata: {file_metadata}")
                
                comp_params = extract_computations_and_parameters(lf, logger)
                logger.debug(f"Computation parameters: {comp_params}")
                
                calib_equip = extract_calibrations_and_equipment(lf, logger)
                logger.debug(f"Calibrations and equipment: {calib_equip}")
                
                zones_data = extract_zones(lf, logger)
                logger.debug(f"Zones data: {zones_data}")
                
                # Process each frame
                frames = lf.frames
                logger.info(f"Found {len(frames)} frames")
                
                for frame_idx, frame in enumerate(frames):
                    try:
                        logger.debug(f"Processing frame {frame_idx + 1}")
                        
                        # Get channels from frame
                        channels = frame.channels
                        logger.debug(f"Found {len(channels)} channels in frame")
                        
                        # Find depth channel
                        depth_ch = find_depth_channel(channels, logger)
                        
                        if not depth_ch:
                            logger.warning(f"No depth channel found in frame {frame_idx + 1}")
                            continue
                        
                        # Extract depth values
                        depth_values = extract_channel_values(depth_ch, logger)
                        logger.debug(f"Extracted {len(depth_values)} depth values")
                        
                        if not depth_values:
                            logger.warning(f"No depth values found in frame {frame_idx + 1}")
                            continue
                        
                        # Process each channel (except depth)
                        for ch in channels:
                            if ch is depth_ch:
                                continue
                            
                            ch_name = safe_get_attr(ch, 'name', '')
                            logger.debug(f"Processing channel: {ch_name}")
                            
                            # Extract channel values
                            values = extract_channel_values(ch, logger)
                            
                            if not values:
                                logger.debug(f"No values found for channel {ch_name}")
                                continue
                            
                            # Extract enhanced tool information
                            tool_info = extract_enhanced_tool_info(ch, logger)
                            
                            # Map to canonical schema with enhanced metadata
                            rec = map_channel_to_canonical(
                                ch_name, values, depth_values, origin_info, 
                                tool_info, geo_info, file_metadata, comp_params, 
                                calib_equip, zones_data, file_info, logger
                            )
                            
                            if rec:
                                records.append(rec)
                                logger.debug(f"Added record for channel {ch_name}")
                    
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_idx + 1}: {e}")
                        continue
    
    except Exception as e:
        logger.error(f"Error loading DLIS file {file_path}: {e}")
        return []
    
    logger.info(f"Total records extracted from {file_path}: {len(records)}")
    return records
