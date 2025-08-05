# parsers/dlis.py

import os
import hashlib
import numpy as np
from datetime import datetime
from dlisio.dlis import load
import logging

# === Exactly 75 canonical fields ===
CANONICAL_FIELDS = [
    "well_id","file_origin","record_type","curve_name",
    "depth_start","depth_end","sample_interval","num_samples",
    "sample_mean","sample_min","sample_max","sample_stddev",
    "facies_code","facies_confidence","horizon_name","horizon_depth",
    "plan_md","plan_tvd","plan_inclination","plan_azimuth",
    "seismic_inline","seismic_xline","seismic_sample_rate","seismic_trace_count",
    "vshale","porosity","water_saturation","permeability",
    "bulk_vol_water","clay_volume",
    "carbonate_flag","coal_flag","sand_flag",
    "archie_a","archie_m","archie_n",
    "water_resistivity","formation_temp","formation_press",
    "core_porosity","core_permeability",
    "tool_type","service_company","processing_software",
    "acquisition_date","processing_date","analyst",
    "qc_flag","null_count",
    "remarks","file_checksum","version",
    "rop","weight_on_bit","torque","pump_pressure",
    "mud_flow_rate","mud_viscosity","mud_weight_actual","caliper",
    "sp_curves","resistivity_deep","resistivity_medium","resistivity_shallow",
    "vp","vs","production_rate","gas_oil_ratio",
    # 7 reserved slots
    "reserved_1","reserved_2","reserved_3","reserved_4",
    "reserved_5","reserved_6","reserved_7"
]

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
        origins = logical_file.origins()
        
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

def extract_tool_info(channel, logger):
    """Extract tool information from channel."""
    tool_info = {
        'tool_type': "",
        'processing_software': "",
        'remarks': ""
    }
    
    try:
        # Get tool information from channel
        tool_info['tool_type'] = safe_get_attr(channel, 'type', "") or ""
        tool_info['processing_software'] = safe_get_attr(channel, 'software', "") or ""
        tool_info['remarks'] = safe_get_attr(channel, 'long_name', "") or ""
        
    except Exception as e:
        logger.debug(f"Error extracting tool info: {e}")
    
    return tool_info

def find_depth_channel(channels, logger):
    """Find the depth channel from available channels."""
    depth_ch = None
    
    # Common depth channel names
    depth_names = ['DEPTH', 'MD', 'TVD', 'TVDSS', 'KB', 'DEPT', 'DEPTH_MD', 'DEPTH_TVD']
    
    for ch in channels:
        ch_name = safe_get_attr(ch, 'name', '').upper()
        if ch_name in depth_names:
            depth_ch = ch
            logger.debug(f"Found depth channel: {ch_name}")
            break
    
    # If no exact match, try partial matches
    if not depth_ch:
        for ch in channels:
            ch_name = safe_get_attr(ch, 'name', '').upper()
            if any(depth_word in ch_name for depth_word in ['DEPTH', 'MD', 'TVD', 'KB']):
                depth_ch = ch
                logger.debug(f"Using {ch_name} as depth channel")
                break
    
    return depth_ch

def compute_statistics(values):
    """Compute statistics for a list of values."""
    if not values:
        return None, None, None, None, 0
    
    arr = np.array(values, dtype=float)
    n = arr.size
    
    if n == 0:
        return None, None, None, None, 0
    
    # Compute stats (handle NaN values)
    mean = float(np.nanmean(arr)) if n else None
    mn = float(np.nanmin(arr)) if n else None
    mx = float(np.nanmax(arr)) if n else None
    std = float(np.nanstd(arr)) if n else None
    
    return mean, mn, mx, std, n

def map_channel_to_canonical(channel_name, values, depth_values, origin_info, tool_info, 
                            file_info, logger):
    """Map channel data to canonical schema."""
    
    # Compute statistics
    mean, mn, mx, std, n = compute_statistics(values)
    
    if n == 0:
        logger.debug(f"No valid data for channel {channel_name}")
        return None
    
    # Calculate depth range
    depth_start = depth_values[0] if depth_values else None
    depth_end = depth_values[-1] if depth_values else None
    sample_interval = (depth_values[1] - depth_values[0]) if len(depth_values) > 1 else None
    
    # Build canonical record
    rec = {field: "" for field in CANONICAL_FIELDS}
    
    rec.update({
        "well_id": origin_info['well_name'],
        "file_origin": file_info['filename'],
        "record_type": "logging",
        "curve_name": channel_name,
        "depth_start": depth_start,
        "depth_end": depth_end,
        "sample_interval": sample_interval,
        "num_samples": n,
        "sample_mean": mean,
        "sample_min": mn,
        "sample_max": mx,
        "sample_stddev": std,
        "service_company": origin_info['service_company'],
        "processing_software": tool_info['processing_software'],
        "acquisition_date": origin_info['acquisition_date'],
        "processing_date": file_info['processing_date'],
        "file_checksum": file_info['checksum'],
        "version": origin_info['version'],
        "tool_type": tool_info['tool_type'],
        "analyst": origin_info['analyst'],
        "remarks": tool_info['remarks'],
        "null_count": len([v for v in values if v is None or np.isnan(v) or v <= -999.0])
    })
    
    # Map specific channel types to canonical fields
    channel_upper = channel_name.upper()
    
    # Petrophysical properties
    if 'POROSITY' in channel_upper:
        rec['porosity'] = mean
    elif 'WATER_SAT' in channel_upper or 'SW' in channel_upper:
        rec['water_saturation'] = mean
    elif 'PERMEABILITY' in channel_upper or 'PERM' in channel_upper:
        rec['permeability'] = mean
    elif 'VSHALE' in channel_upper or 'VSH' in channel_upper:
        rec['vshale'] = mean
    elif 'CLAY' in channel_upper:
        rec['clay_volume'] = mean
    elif 'BULK_VOL_WATER' in channel_upper:
        rec['bulk_vol_water'] = mean
    
    # Resistivity measurements
    elif 'RESISTIVITY' in channel_upper:
        if 'DEEP' in channel_upper:
            rec['resistivity_deep'] = mean
        elif 'MEDIUM' in channel_upper or 'MED' in channel_upper:
            rec['resistivity_medium'] = mean
        elif 'SHALLOW' in channel_upper or 'SHA' in channel_upper:
            rec['resistivity_shallow'] = mean
        else:
            rec['resistivity_deep'] = mean  # Default to deep
    
    # Drilling parameters
    elif 'ROP' in channel_upper:
        rec['rop'] = mean
    elif 'WOB' in channel_upper or 'WEIGHT_ON_BIT' in channel_upper:
        rec['weight_on_bit'] = mean
    elif 'TORQUE' in channel_upper:
        rec['torque'] = mean
    elif 'PUMP_PRESSURE' in channel_upper:
        rec['pump_pressure'] = mean
    
    # Mud properties
    elif 'MUD_FLOW' in channel_upper:
        rec['mud_flow_rate'] = mean
    elif 'MUD_VISCOSITY' in channel_upper:
        rec['mud_viscosity'] = mean
    elif 'MUD_WEIGHT' in channel_upper:
        rec['mud_weight_actual'] = mean
    
    # Logging measurements
    elif 'CALIPER' in channel_upper:
        rec['caliper'] = mean
    elif 'SP' in channel_upper or 'SPONTANEOUS' in channel_upper:
        rec['sp_curves'] = mean
    
    # Acoustic properties
    elif 'VP' in channel_upper or 'P_WAVE' in channel_upper:
        rec['vp'] = mean
    elif 'VS' in channel_upper or 'S_WAVE' in channel_upper:
        rec['vs'] = mean
    
    # Formation conditions
    elif 'FORMATION_TEMP' in channel_upper or 'TEMP' in channel_upper:
        rec['formation_temp'] = mean
    elif 'FORMATION_PRESS' in channel_upper or 'PRESSURE' in channel_upper:
        rec['formation_press'] = mean
    
    # Archie parameters
    elif 'ARCHIE_A' in channel_upper:
        rec['archie_a'] = mean
    elif 'ARCHIE_M' in channel_upper:
        rec['archie_m'] = mean
    elif 'ARCHIE_N' in channel_upper:
        rec['archie_n'] = mean
    elif 'WATER_RESISTIVITY' in channel_upper or 'RW' in channel_upper:
        rec['water_resistivity'] = mean
    
    # Production data
    elif 'PRODUCTION_RATE' in channel_upper:
        rec['production_rate'] = mean
    elif 'GAS_OIL_RATIO' in channel_upper or 'GOR' in channel_upper:
        rec['gas_oil_ratio'] = mean
    
    return rec

def parse_dlis(file_path: str, logger: logging.Logger) -> list[dict]:
    """
    Comprehensive DLIS parser: extract all available data from DLIS files
    and map to the 75-field canonical schema.
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
                
                # Extract origin information
                origin_info = extract_origin_info(lf, logger)
                logger.debug(f"Origin info: {origin_info}")
                
                # Process each frame
                frames = lf.frames()
                logger.info(f"Found {len(frames)} frames")
                
                for frame_idx, frame in enumerate(frames):
                    try:
                        logger.debug(f"Processing frame {frame_idx + 1}")
                        
                        # Get channels from frame
                        channels = frame.channels()
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
                            
                            # Extract tool information
                            tool_info = extract_tool_info(ch, logger)
                            
                            # Map to canonical schema
                            rec = map_channel_to_canonical(
                                ch_name, values, depth_values, origin_info, 
                                tool_info, file_info, logger
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
