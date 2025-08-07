import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from schema.fields import CANONICAL_FIELDS

def safe_get_attr(obj, attr, nested_attr=None, default=None):
    """Safely get attribute value with fallback."""
    try:
        if nested_attr:
            # Handle nested attribute access (e.g., obj.WELL.value)
            # First try direct access for well section items
            if hasattr(obj, '__iter__'):
                for item in obj:
                    if hasattr(item, 'mnemonic') and item.mnemonic == attr:
                        result = getattr(item, nested_attr, default)
                        return result
            
            # Fallback to get() method
            attr_obj = obj.get(attr, None)
            if attr_obj is not None:
                result = getattr(attr_obj, nested_attr, default)
                return result
            return default
        else:
            # For SectionItems, use get() method
            return obj.get(attr, default)
    except Exception as e:
        print(f"Error in safe_get_attr: {e}")
        return default

def dms_to_decimal(dms_str):
    """Convert DMS (Degrees, Minutes, Seconds) format to decimal degrees."""
    if not dms_str or not isinstance(dms_str, str):
        return None
    
    try:
        # Remove any extra spaces and split
        parts = dms_str.strip().split()
        if len(parts) < 3:
            return None
            
        # Extract degrees, minutes, seconds
        degrees = float(parts[0])
        minutes = float(parts[1].replace("'", ""))
        seconds = float(parts[2].replace('"', ""))
        
        # Calculate decimal degrees
        decimal_degrees = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        # Handle direction (N/S, E/W)
        if len(parts) > 3:
            direction = parts[3]
            if direction in ['S', 'W']:
                decimal_degrees = -decimal_degrees
                
        return decimal_degrees
        
    except (ValueError, IndexError):
        return None

def extract_well_info(las_file, logger):
    """Extract well information from LAS file header."""
    well_info = {
        'well_id': "",
        'field_name': "",
        'country': "",
        'state_province': "",
        'service_company': "",
        'acquisition_date': None,
        'api_number': "",
        'uwi': "",
        'location': "",
        'start_depth': None,
        'stop_depth': None,
        'step_size': None,
        'latitude': None,
        'longitude': None,
        'elevation': None,
        'county': "",  # Will go to remarks if not in canonical
        'remarks': []  # For metadata that doesn't fit canonical fields
    }
    
    try:
        # Extract well header information
        well_info['well_id'] = safe_get_attr(las_file.well, 'WELL', 'value', "") or ""
        well_info['field_name'] = safe_get_attr(las_file.well, 'FLD', 'value', "") or ""
        well_info['country'] = safe_get_attr(las_file.well, 'CTRY', 'value', "") or ""
        well_info['state_province'] = safe_get_attr(las_file.well, 'STAT', 'value', "") or ""
        well_info['service_company'] = safe_get_attr(las_file.well, 'SRVC', 'value', "") or ""
        well_info['api_number'] = safe_get_attr(las_file.well, 'API', 'value', "") or ""
        well_info['uwi'] = safe_get_attr(las_file.well, 'UWI', 'value', "") or ""
        well_info['location'] = safe_get_attr(las_file.well, 'LOC', 'value', "") or ""
        
        # Extract depth information
        start_depth = safe_get_attr(las_file.well, 'STRT', 'value', None)
        stop_depth = safe_get_attr(las_file.well, 'STOP', 'value', None)
        step_size = safe_get_attr(las_file.well, 'STEP', 'value', None)
        
        # Convert to float if they exist
        if start_depth:
            well_info['start_depth'] = float(start_depth)
        if stop_depth:
            well_info['stop_depth'] = float(stop_depth)
        if step_size:
            well_info['step_size'] = float(step_size)
        
        # Extract geographic data
        lati = safe_get_attr(las_file.well, 'LATI', 'value', None)
        long = safe_get_attr(las_file.well, 'LONG', 'value', None)
        elev = safe_get_attr(las_file.well, 'ELEV', 'value', None)
        
        # Convert DMS to decimal degrees
        if lati:
            well_info['latitude'] = dms_to_decimal(lati)
        if long:
            well_info['longitude'] = dms_to_decimal(long)
        if elev:
            try:
                well_info['elevation'] = float(elev)
            except (ValueError, TypeError):
                pass
        
        # Extract county (will go to remarks)
        county = safe_get_attr(las_file.well, 'CNTY', 'value', None)
        if county:
            well_info['county'] = county
        
        # Extract date
        date_str = safe_get_attr(las_file.well, 'DATE', None)
        if date_str:
            try:
                # Parse LAS date format (DD-MMM-YYYY)
                from datetime import datetime
                well_info['acquisition_date'] = datetime.strptime(date_str, '%d-%b-%Y').isoformat()
            except:
                well_info['acquisition_date'] = date_str
                
    except Exception as e:
        logger.debug(f"Error extracting well info: {e}")
    
    return well_info

def extract_parameter_info(las_file, logger):
    """Extract parameter information from LAS file."""
    parameters = {
        'archie_a': None,
        'archie_m': None,
        'archie_n': None,
        'water_resistivity': None,
        'mud_weight_actual': None,
        'mud_viscosity': None,
        'mud_flow_rate': None,
        'rop': None,
        'weight_on_bit': None,
        'torque': None,
        'pump_pressure': None,
        'remarks': []  # For metadata that doesn't fit canonical fields
    }
    
    try:
        if hasattr(las_file, 'params') and las_file.params:
            for item in las_file.params:
                if hasattr(item, 'mnemonic') and hasattr(item, 'value'):
                    mnemonic = item.mnemonic.upper()
                    value = item.value
                    
                    # Archie parameters
                    if mnemonic == 'A':
                        parameters['archie_a'] = float(value) if value else None
                    elif mnemonic == 'M':
                        parameters['archie_m'] = float(value) if value else None
                    elif mnemonic == 'N':
                        parameters['archie_n'] = float(value) if value else None
                    elif mnemonic in ['RW', 'RMF']:
                        parameters['water_resistivity'] = float(value) if value else None
                    
                    # Mud properties
                    elif mnemonic in ['MW', 'MUDWT']:
                        parameters['mud_weight_actual'] = float(value) if value else None
                    elif mnemonic == 'MUDVISC':
                        parameters['mud_viscosity'] = float(value) if value else None
                    elif mnemonic == 'MUD_FLOW':
                        parameters['mud_flow_rate'] = float(value) if value else None
                    
                    # Drilling parameters
                    elif mnemonic == 'ROP':
                        parameters['rop'] = float(value) if value else None
                    elif mnemonic == 'WOB':
                        parameters['weight_on_bit'] = float(value) if value else None
                    elif mnemonic == 'TORQUE':
                        parameters['torque'] = float(value) if value else None
                    elif mnemonic == 'PUMP_PRESS':
                        parameters['pump_pressure'] = float(value) if value else None
                    
                    # Add other parameters to remarks
                    else:
                        unit = getattr(item, 'unit', '')
                        descr = getattr(item, 'descr', '')
                        param_info = f"{mnemonic}: {value}"
                        if unit:
                            param_info += f" {unit}"
                        if descr:
                            param_info += f" ({descr})"
                        parameters['remarks'].append(param_info)
                        
    except Exception as e:
        logger.debug(f"Error extracting parameter info: {e}")
    
    return parameters

def extract_version_info(las_file, logger):
    """Extract version information from LAS file."""
    version_info = {
        'version': None,
        'wrap': None,
        'prog': None,
        'remarks': []
    }
    
    try:
        if hasattr(las_file, 'version') and las_file.version:
            for item in las_file.version:
                if hasattr(item, 'mnemonic') and hasattr(item, 'value'):
                    mnemonic = item.mnemonic.upper()
                    value = item.value
                    
                    if mnemonic == 'VER':
                        version_info['version'] = value
                    elif mnemonic == 'WRAP':
                        version_info['wrap'] = value
                    elif mnemonic == 'PROG':
                        version_info['prog'] = value
                    else:
                        version_info['remarks'].append(f"{mnemonic}: {value}")
                        
    except Exception as e:
        logger.debug(f"Error extracting version info: {e}")
    
    return version_info

def extract_curve_data(las_file, logger):
    """Extract curve data and statistics from LAS file."""
    curves_data = []
    
    try:
        for curve_item in las_file.curves:
            curve_name = curve_item.mnemonic
            if curve_name == 'DEPTH':
                continue  # Skip depth curve as it's handled separately
                
            curve_data = curve_item.data
            if curve_data is None or len(curve_data) == 0:
                continue
                
            # Calculate statistics
            null_item = las_file.well.get('NULL')
            null_value = null_item.value if null_item else -999.25
            valid_data = curve_data[curve_data != null_value]
            if len(valid_data) == 0:
                continue
                
            mean_val = float(np.mean(valid_data))
            min_val = float(np.min(valid_data))
            max_val = float(np.max(valid_data))
            
            # Extract curve metadata
            unit = getattr(curve_item, 'unit', '')
            descr = getattr(curve_item, 'descr', '')
            api_code = getattr(curve_item, 'api_code', '')
            
            curves_data.append({
                'curve_name': curve_name,
                'values': curve_data.tolist(),
                'mean': mean_val,
                'min': min_val,
                'max': max_val,
                'count': len(valid_data),
                'unit': unit,
                'descr': descr,
                'api_code': api_code
            })
            
    except Exception as e:
        logger.debug(f"Error extracting curve data: {e}")
    
    return curves_data

def map_curve_to_canonical(curve_name, curve_data, well_info, param_info, version_info, file_info, logger):
    """Map LAS curve data to canonical schema."""
    rec = {field: "" for field in CANONICAL_FIELDS}
    
    # Set hard-coded defaults for fields with no extraction logic
    rec.update({
        "facies_code": "UNKNOWN",
        "horizon_name": "UNKNOWN"
    })
    
    # Basic well information
    rec.update({
        "well_id": well_info['well_id'],
        "field_name": well_info['field_name'],
        "country": well_info['country'],
        "state_province": well_info['state_province'],
        "service_company": well_info['service_company'],
        "acquisition_date": well_info['acquisition_date'],
        "record_type": "LAS_LOG",
        "curve_name": curve_name,
        "file_origin": file_info['file_path'],
        "depth_start": well_info['start_depth'],
        "depth_end": well_info['stop_depth'],
        "tool_type": "LAS_LOG"
    })
    
    # Add non-canonical fields to remarks
    if 'remarks' not in rec:
        rec['remarks'] = ""
    rec['remarks'] += f"api_number: {well_info['api_number'] or '(not found)'}; "
    rec['remarks'] += f"uwi: {well_info['uwi'] or '(not found)'}; "
    rec['remarks'] += f"location: {well_info['location'] or '(not found)'}; "
    rec['remarks'] += f"step_size: {well_info['step_size'] or '(not found)'}; "
    
    # Geographic data
    if well_info['latitude'] is not None:
        rec['latitude'] = well_info['latitude']
    if well_info['longitude'] is not None:
        rec['longitude'] = well_info['longitude']
    if well_info['elevation'] is not None:
        rec['elevation'] = well_info['elevation']
    
    # Parameter data (Archie, mud, drilling)
    if param_info['archie_a'] is not None:
        rec['archie_a'] = param_info['archie_a']
    if param_info['archie_m'] is not None:
        rec['archie_m'] = param_info['archie_m']
    if param_info['archie_n'] is not None:
        rec['archie_n'] = param_info['archie_n']
    if param_info['water_resistivity'] is not None:
        rec['water_resistivity'] = param_info['water_resistivity']
    if param_info['mud_weight_actual'] is not None:
        rec['mud_weight_actual'] = param_info['mud_weight_actual']
    if param_info['mud_viscosity'] is not None:
        rec['mud_viscosity'] = param_info['mud_viscosity']
    if param_info['mud_flow_rate'] is not None:
        rec['mud_flow_rate'] = param_info['mud_flow_rate']
    if param_info['rop'] is not None:
        rec['rop'] = param_info['rop']
    if param_info['weight_on_bit'] is not None:
        rec['weight_on_bit'] = param_info['weight_on_bit']
    if param_info['torque'] is not None:
        rec['torque'] = param_info['torque']
    if param_info['pump_pressure'] is not None:
        rec['pump_pressure'] = param_info['pump_pressure']
    
    # Version information
    if version_info['version']:
        rec['version'] = version_info['version']
    
    # Map curve data based on curve name
    curve_upper = curve_name.upper()
    mean = curve_data['mean']
    
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
    if 'mean' in curve_data:
        rec['sample_mean'] = curve_data['mean']
    if 'min' in curve_data:
        rec['sample_min'] = curve_data['min']
    if 'max' in curve_data:
        rec['sample_max'] = curve_data['max']
    if 'count' in curve_data:
        rec['num_samples'] = curve_data['count']
    
    # Build remarks field with metadata that doesn't fit canonical fields
    remarks_parts = []
    
    # Add county if available
    if well_info['county']:
        remarks_parts.append(f"County: {well_info['county']}")
    
    # Add version info
    if version_info['wrap']:
        remarks_parts.append(f"Wrap: {version_info['wrap']}")
    if version_info['prog']:
        remarks_parts.append(f"Program: {version_info['prog']}")
    
    # Add parameter remarks
    if param_info['remarks']:
        remarks_parts.extend(param_info['remarks'])
    
    # Add version remarks
    if version_info['remarks']:
        remarks_parts.extend(version_info['remarks'])
    
    # Add curve metadata
    if curve_data.get('unit'):
        remarks_parts.append(f"Unit: {curve_data['unit']}")
    if curve_data.get('descr'):
        remarks_parts.append(f"Description: {curve_data['descr']}")
    if curve_data.get('api_code'):
        remarks_parts.append(f"API Code: {curve_data['api_code']}")
    
    # Combine all remarks, preserving existing non-canonical fields
    if remarks_parts:
        existing_remarks = rec.get('remarks', '')
        new_remarks = "; ".join(remarks_parts)
        rec['remarks'] = existing_remarks + new_remarks if existing_remarks else new_remarks
    
    # Set conditional defaults for fields with extraction logic
    if not rec.get('production_rate'):
        rec['production_rate'] = 0.0
    if not rec.get('gas_oil_ratio'):
        rec['gas_oil_ratio'] = 0.0
    if not rec.get('latitude') or rec.get('latitude') is None:
        rec['latitude'] = 0.0
    if not rec.get('longitude') or rec.get('longitude') is None:
        rec['longitude'] = 0.0
    
    return rec

def parse_las(file_path, logger):
    """Parse LAS file and return list of canonical records."""
    try:
        import lasio
        
        # Load LAS file
        las_file = lasio.read(file_path)
        logger.info(f"Successfully loaded LAS file: {file_path}")
        
        # Extract all information
        well_info = extract_well_info(las_file, logger)
        param_info = extract_parameter_info(las_file, logger)
        version_info = extract_version_info(las_file, logger)
        curves_data = extract_curve_data(las_file, logger)
        
        # Create canonical records
        records = []
        file_info = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'file_type': 'LAS'
        }
        
        for curve_data in curves_data:
            record = map_curve_to_canonical(
                curve_data['curve_name'],
                curve_data,
                well_info,
                param_info,
                version_info,
                file_info,
                logger
            )
            records.append(record)
        
        logger.info(f"Generated {len(records)} records from LAS file")
        return records
        
    except Exception as e:
        logger.error(f"Error parsing LAS file {file_path}: {e}")
        return [] 