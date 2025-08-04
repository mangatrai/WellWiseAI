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
    """Extract values from channel using the correct dlisio API."""
    channel_name = safe_get_attr(channel, 'name', 'UNKNOWN')
    logger.debug(f"DEBUG: Extracting values for channel '{channel_name}'")
    
    try:
        # Use the correct dlisio API method
        channel_curves = channel.curves()
        
        if channel_curves is not None:
            # Convert numpy array to list of floats, filtering out null values
            values = []
            for val in channel_curves:
                # Skip null values (typically -999.25 or similar)
                if val is not None and not np.isnan(val) and val > -999.0:
                    values.append(float(val))
            
            logger.debug(f"DEBUG: Successfully extracted {len(values)} valid values from '{channel_name}'")
            logger.debug(f"DEBUG: First 5 values: {values[:5] if len(values) >= 5 else values}")
            return values
        else:
            logger.debug(f"DEBUG: Channel '{channel_name}' curves() returned None")
            return []
            
    except Exception as e:
        logger.debug(f"DEBUG: Failed to extract values from '{channel_name}': {e}")
        return []

def extract_origin_info(logical_file, logger):
    """Extract origin information from logical file."""
    origin_info = {
        'well_name': "",
        'company': "",
        'acquisition_date': "",
        'version': ""
    }
    
    try:
        origins = safe_get_attr(logical_file, 'origins', [])
        if origins and len(origins) > 0:
            origin = origins[0]
            origin_info['well_name'] = safe_get_attr(origin, 'well_name', "") or ""
            origin_info['company'] = safe_get_attr(origin, 'company', "") or ""
            origin_info['version'] = safe_get_attr(origin, 'version', "") or ""
            
            # Handle acquisition date
            ct = safe_get_attr(origin, 'creation_time', "")
            if ct and hasattr(ct, 'isoformat'):
                origin_info['acquisition_date'] = ct.isoformat()
            elif ct:
                origin_info['acquisition_date'] = str(ct)
    except Exception as e:
        logger.warning(f"Error extracting origin info: {e}")
    
    return origin_info

def parse_dlis(file_path: str, logger: logging.Logger) -> list[dict]:
    """
    Robust DLIS parser: extract depth and channel values using DLISIO, compute stats,
    and map into the 75-field canonical schema.
    """
    logger.debug(f"Starting to parse {file_path}")
    
    fname = os.path.basename(file_path)
    checksum = compute_checksum(file_path)
    processing_date = datetime.utcnow().isoformat()
    
    records = []
    
    try:
        with load(str(file_path)) as logical_files:
            logger.debug(f"Found {len(logical_files)} logical files")
            
            for lf_idx, lf in enumerate(logical_files):
                logger.debug(f"Processing logical file {lf_idx + 1}")
                
                # Extract origin information
                origin_info = extract_origin_info(lf, logger)
                logger.debug(f"Origin info: {origin_info}")
                
                # Process each frame
                frames = safe_get_attr(lf, 'frames', [])
                logger.debug(f"Found {len(frames)} frames")
                
                for frame_idx, frame in enumerate(frames):
                    try:
                        logger.debug(f"Processing frame {frame_idx + 1}")
                        
                        # Find the DEPTH channel
                        depth_ch = None
                        channels = safe_get_attr(frame, 'channels', [])
                        logger.debug(f"Found {len(channels)} channels in frame")
                        
                        # Log all channel names for debugging
                        channel_names = []
                        for ch in channels:
                            ch_name = safe_get_attr(ch, 'name', '')
                            channel_names.append(ch_name)
                            logger.debug(f"Channel: {ch_name}")
                        
                        for ch in channels:
                            ch_name = safe_get_attr(ch, 'name', '')
                            if isinstance(ch_name, str) and ch_name.strip().upper() == 'DEPTH':
                                depth_ch = ch
                                logger.debug(f"Found DEPTH channel: {ch_name}")
                                break
                        
                        if not depth_ch:
                            logger.debug(f"No DEPTH channel found, trying alternative depth channels")
                            # Try to find any channel that might be depth-related
                            for ch in channels:
                                ch_name = safe_get_attr(ch, 'name', '').upper()
                                if any(depth_word in ch_name for depth_word in ['MD', 'TVD', 'TVDSS', 'KB']):
                                    depth_ch = ch
                                    logger.debug(f"Using {ch_name} as depth channel")
                                    break
                            if not depth_ch:
                                logger.debug(f"No depth channel found, skipping frame")
                                continue
                        
                        # Extract depth values
                        depths = extract_channel_values(depth_ch, logger)
                        logger.debug(f"Extracted {len(depths)} depth values")
                        if not depths:
                            logger.debug(f"No depth values found")
                            continue
                        
                        # Process each channel (except depth)
                        for ch in channels:
                            if ch is depth_ch:
                                continue
                            
                            ch_name = safe_get_attr(ch, 'name', '')
                            logger.debug(f"Processing channel: {ch_name}")
                            
                            # Extract channel values
                            values = extract_channel_values(ch, logger)
                            logger.debug(f"Extracted {len(values)} values for channel {ch_name}")
                            
                            if not values:
                                logger.debug(f"No values found for channel {ch_name}")
                                continue
                            
                            # Compute statistics
                            arr = np.array(values, dtype=float)
                            n = arr.size
                            
                            # Skip if no valid data
                            if n == 0:
                                logger.debug(f"No valid data for channel {ch_name}")
                                continue
                            
                            # Compute stats (handle NaN values)
                            mean = float(np.nanmean(arr)) if n else None
                            mn = float(np.nanmin(arr)) if n else None
                            mx = float(np.nanmax(arr)) if n else None
                            std = float(np.nanstd(arr)) if n else None
                            
                            logger.debug(f"Stats for {ch_name}: mean={mean}, min={mn}, max={mx}, std={std}")
                            
                            # Build the canonical record
                            rec = {c: "" for c in CANONICAL_FIELDS}  # Use empty strings instead of None
                            rec.update({
                                "well_id": origin_info['well_name'] or "",
                                "file_origin": fname,
                                "record_type": "log_curve",
                                "curve_name": (safe_get_attr(ch, "name", "") or safe_get_attr(ch, "mnemonic", "")) or "",
                                "depth_start": depths[0] if depths else "",
                                "depth_end": depths[-1] if depths else "",
                                "sample_interval": (depths[1] - depths[0]) if len(depths) > 1 else "",
                                "num_samples": int(n),
                                "sample_mean": mean or "",
                                "sample_min": mn or "",
                                "sample_max": mx or "",
                                "sample_stddev": std or "",
                                "service_company": origin_info['company'] or "",
                                "processing_software": "",
                                "acquisition_date": origin_info['acquisition_date'] or "",
                                "processing_date": processing_date,
                                "file_checksum": checksum,
                                "version": origin_info['version'] or "",
                                "tool_type": safe_get_attr(ch, "type", "") or "",
                                "analyst": "",
                                "remarks": safe_get_attr(ch, "long_name", "") or "",
                            })
                            
                            records.append(rec)
                            logger.debug(f"Added record for channel {ch_name}")
                    
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_idx + 1}: {e}")
                        continue
    
    except Exception as e:
        logger.error(f"Error loading DLIS file {file_path}: {e}")
        return []
    
    logger.debug(f"Total records extracted from {file_path}: {len(records)}")
    return records
