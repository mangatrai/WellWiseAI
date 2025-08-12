The Canonical Schema Discussion Points:
density field - Legitimate bulk density data (2.44-2.50 g/cm³) from RHOB curves
Missing petrophysical fields - We're finding real measurements that don't fit the current schema
Field mapping strategy - Should we expand the schema or be more aggressive about moving to remarks?

Complete List of Fields Mapped to remarks in Both Parsers (DLIS/LAS):
LAS Parser (parsers/las.py):
1. Non-canonical well information:
api_number - API number (or "(not found)")
uwi - Unique Well Identifier (or "(not found)")
location - Well location (or "(not found)")
step_size - Step size (or "(not found)")
2. Curve-specific data:
gamma_ray - Gamma ray measurements (from GR/GAMMA curves)
sonic_transit_time - Sonic transit time (from DT/SONIC curves)
density - Bulk density measurements (from RHOB curves)
neutron_porosity - Neutron porosity measurements (from NPHI curves)
photoelectric_factor - Photoelectric factor measurements (from PEF curves)
3. Metadata from well info:
county - County information
wrap - Wrap information
prog - Program information
4. Parameter remarks:
Various parameter information from param_info['remarks']
5. Version remarks:
Various version information from version_info['remarks']
6. Curve metadata:
unit - Curve units
descr - Curve description
api_code - API code
DLIS Parser (parsers/dlis.py):
1. Non-canonical well information (same as LAS):
api_number - "(not extracted from DLIS)"
uwi - "(not extracted from DLIS)"
location - "(not extracted from DLIS)"
step_size - "(not extracted from DLIS)"
2. Curve-specific data (same as LAS):
gamma_ray - Gamma ray measurements (from GR/GAMMA curves)
sonic_transit_time - Sonic transit time (from DT/SONIC curves)
density - Bulk density measurements (from RHOB curves)
neutron_porosity - Neutron porosity measurements (from NPHI curves)
photoelectric_factor - Photoelectric factor measurements (from PEF curves)
3. File metadata:
job_id - Job ID
client_name - Client name
formation - Formation name
log_date - Log date
software - Software used
comments - Comments
4. Computation parameters:
Various computation parameters from comp_params['remarks']
5. Calibration/Equipment data:
Calibration and equipment information from calib_equip['remarks']
6. Zones data:
Zone information from zones_data['remarks']
7. Curve metadata:
curve_units - Curve units
curve_description - Curve description
curve_type - Curve type
Summary:
Both parsers map similar core fields to remarks, with DLIS having additional metadata fields due to the richer structure of DLIS files. The key non-canonical fields being moved to remarks are:
api_number, uwi, location, step_size
gamma_ray, sonic_transit_time, density, neutron_porosity, photoelectric_factor
Various metadata and parameter information

CSV Parser (parsers/csv.py) - PLANNED:
1. Direct Canonical Mappings (39 fields):
depth_start - Measured Depth m
plan_tvd - Hole Depth (TVD) m
rop - Rate of Penetration m/h
weight_on_bit - Weight on Bit kkgf
torque - Average Surface Torque kN.m
sample_mean - Total Downhole RPM rpm
sample_stddev - Bit run number unitless
sp_curves - ARC Gamma Ray (BH corrected) gAPI
resistivity_deep - ARC Phase Shift Resistivity 40 inch at 2 MHz ohm.m
resistivity_medium - ARC Phase Shift Resistivity 28 inch at 2 MHz ohm.m
resistivity_shallow - ARC Attenuation Resistivity 28 inch at 2 MHz ohm.m
mud_flow_rate - Mud Flow In L/min
mud_weight_actual - Mud Density In g/cm3
pump_pressure - Average Standpipe Pressure kPa
formation_temp - Annular Temperature degC
formation_press - ARC Annular Pressure kPa
porosity - Density Porosity from ROBB_RT m3/m3
caliper - Utrasonic Caliper, Average Diameter, Computed DH cm
vshale - Bulk Density, Bottom, Computed DH g/cm3
water_resistivity - ARC Phase Shift Resistivity 40 inch at 2 MHz ohm.m
water_saturation - Density Porosity from ROBB_RT m3/m3
gas_oil_ratio - Gas (avg) %
production_rate - Mud Flow In L/min
qc_flag - MWD Shock Risk unitless
sample_interval - Total SPM 1/min
num_samples - Pump Time h
permeability - Corr. Drilling Exponent unitless
field_name - nameWellbore
service_company - name
tool_type - Pass Name unitless
file_origin - Bit run number unitless
archie_a - ARC Phase Shift Resistivity 40 inch at 2 MHz ohm.m
archie_m - ARC Phase Shift Resistivity 28 inch at 2 MHz ohm.m
core_permeability - Corr. Drilling Exponent unitless
seismic_sample_rate - MWD Vibration X-Axis ft/s2
seismic_trace_count - MWD Vibration Lateral ft/s2
vp - MWD Vibration Torsional kN.m
country - nameWellbore
state_province - name

2. Fields Going to Remarks (145 fields):
Drilling Parameters:
- Average Surface Torque kN.m
- Bit Drilling Run m
- Bit Drilling Time h
- Bit Depth m
- Weight on Bit kkgf
- Total Hookload kkgf
- Average Hookload kkgf
- Total Downhole RPM rpm
- Average Rotary Speed rpm
- Averaged RPM rpm
- MWD Turbine RPM rpm
- MWD Collar RPM rpm
- Rate of Penetration (5ft avg) m/h
- Rate of penetration m/h
- Inverse ROP s/m
- Averaged WOB kkgf
- Bit Revolutions (cum) unitless
- Bit Drill Time h
- Weight On Hook kkgf
- HKLI kkgf
- HKLO kkgf
- String weight (rot, avg) kkgf
- Extrapolated Hole TVD m
- Lag Depth (TVD) m
- Total Vertical Depth m
- Hole depth (MD) m

Shock and Vibration Data:
- CRS Stick-Slip Frequency unitless
- PowerUP Shock Rate 1/s
- CRS Shock Level unitless
- MWD Shock Risk unitless
- MWD Shock Peak m/s2
- MWD Total Shocks unitless
- Isonic Shock, Real-Time unitless
- CRS Stick-Slip Amplitude rpm
- MWD Stick-Slip PKtoPK RPM rpm
- Shock Level, Computed DH unitless
- SHKRSK_P unitless

Pressure and Flow Data:
- S2AC kPa
- S1AC kPa
- Stand Pipe Pressure kPa
- Flow Pumps L/min
- Pump 2 Strokes unitless
- Pump 3 Stroke Rate 1/min
- Pump 4 Stroke Rate 1/min
- Pump 1 Stroke Rate 1/min
- Pump 4 Strokes unitless
- Pump 3 Strokes unitless
- Pump 1 Strokes unitless
- Average Standpipe Pressure kPa
- ARC Annular Pressure kPa
- Downhole Annulus Pressure, Computed DH kPa
- FPWD Fracture Pressure Gradient g/cm3
- FPWD Pore Pressure Gradient g/cm3
- HYDR_RET_P kPa

Temperature Data:
- Annular Temperature degC
- Temperature Out degC
- Downhole Annulus Temperature, Computed DH degC
- TMP In degC
- GTEMP degC
- HSTEMP degC
- MWD DNI Temperature degC

Mud Properties:
- Mud Density In g/cm3
- Mud Density Out g/cm3
- ARC Equivalent Circulating Density g/cm3
- Equivalent Circulating Density g/cm3
- ECD_P g/cm3
- IMWT g/cm3

Formation Evaluation:
- ARC Gamma Ray (uncorrected) gAPI
- Gamma Ray, Average gAPI
- Gamma Ray, Average, Computed DH gAPI
- Density Porosity from ROBB_RT m3/m3
- Best Thermal Neutron Porosity, Average m3/m3
- Thermal Neutron Porosity, Average m3/m3
- Utrasonic Caliper, Average Diameter, Computed DH cm
- Bulk Density, Bottom, Computed DH g/cm3
- Bulk Density Correction, Bottom, Computed DH g/cm3

Resistivity Measurements:
- IMP/ARC Non-BHcorr Phase-Shift Resistivity 40-in. at 2 MHz ohm.m
- IMP/ARC Phase-Shift Conductivity 28-in. at 2 MHz mS/m
- IMP/ARC Phase-Shift Conductivity 40-in. at 2 MHz mS/m
- IMP/ARC Non-BHcorr Attenuation Resistivity 28-in. at 2 MHz ohm.m
- IMP/ARC Attenuation Conductivity 40-in. at 2 MHz mS/m
- IMP/ARC Non-BHcorr Attenuation Resistivity 40-in. at 2 MHz ohm.m
- IMP/ARC Non-BHcorr Phase-Shift Resistivity 28-in. at 2 MHz ohm.m
- IMP/ARC Phase-Shift Resistivity 28-in. at 2 MHz ohm.m
- IMP/ARC Attenuation Resistivity 28-in. at 2 MHz ohm.m
- IMP/ARC Attenuation Resistivity 40-in. at 2 MHz ohm.m
- ARC Phase Shift Resistivity 40 inch at 2 MHz ohm.m
- ARC Phase Shift Resistivity 28 inch at 2 MHz ohm.m
- ARC Attenuation Resistivity 28 inch at 2 MHz ohm.m
- ARC Attenuation Resistivity 40 inch at 2 MHz ohm.m
- ARC Uncorrected Phase-Shift Conductivity 40 inch at 2 MHz, Computed DH mS/m
- ARC Uncorrected Attenuation Resistivity 28 inch at 2 MHz, Computed DH ohm.m
- ARC Uncorrected Attenuation Conductivity 40 inch Spacing at 2 MHz, Computed DH mS/m
- ARC Uncorrected Phase-Shift Conductivity 28 inch at 2 MHz, Computed DH mS/m
- ARC Uncorrected Phase Shift Resistivity 40 inch at 2 MHz, Computed DH ohm.m
- ARC Uncorrected Attenuation Resistivity 40 inch at 2 MHz, Computed DH ohm.m
- ARC Uncorrected Phase Shift Resistivity 28 inch at 2 MHz, Computed DH ohm.m

Gas Chromatography:
- Propane (C3) ppm
- Iso-pentane (IC5) ppm
- n-Penthane ppm
- Ethane (C2) ppm
- Nor-butane (NC4) ppm
- Methane (C1) ppm
- Iso-butane (IC4) ppm
- Gas (avg) %

MWD/LWD Directional Data:
- CRS ToolFace dega
- MWD Gravity Toolface dega
- CRS Desired ToolFace dega
- MWD Continuous Inclination dega
- CRS Continuous Inclination dega
- CRS Continuous Azimuth dega
- MWD Continuous Azimuth dega
- CRS Real-Time Status unitless
- CRS Real-Time Mode unitless
- Rig Mode unitless
- CRS5 unitless
- CRS4 unitless
- CRS1 unitless
- CRS Turbine RPM rpm
- CRS Steering Ratio %

Time and Interval Data:
- Elapsed time in-slips s
- TOFB s
- OSTM s
- TOBO s
- Total SPM 1/min
- Pump Time h
- SHK3TM_RT min
- Transit Time for Pump Off, Real-Time us
- Delta-T Pump Off, Real-Time us/ft
- Delta-T Compressional, Real-Time us/ft

Statistical and Sample Data:
- Total Strokes unitless
- BHFG unitless
- STKSLP unitless
- RHX unitless
- NRPM_RT unitless
- EDRT unitless
- RGX_RT unitless
- AJAM_MWD unitless
- DRET unitless
- RHX_RT unitless
- STUCK_RT unitless
- TNPH_UNC_ECO_RT unitless
- BPHI_UNC_ECO_RT unitless
- UTSTAT unitless
- STWD_RT unitless

Survey and Geophysical Data:
- Survey raw mag transverse nT
- Survey raw grav inv transverse m/s2
- Survey raw mag axial nT
- Survey raw grav transax m/s2
- Survey raw grav axial m/s2
- ANGXCRS dega
- ANGLX dega

SPR MWD Data:
- SPR MWD_01 mwd unitless
- SPR MWD_02 mwd unitless
- SPR MWD_06 mwd unitless
- SPR MWD_04 mwd unitless
- SPR MWD_07 mwd unitless
- SPR MWD_08 mwd unitless
- SPR MWD_03 mwd unitless
- SPR MWD_10 mwd unitless
- SPR MWD_05 mwd unitless

Neutron and Thermal Data:
- Thermal Neutron Far Count Rates, Average, Computed DH 1/s
- Thermal Neutron Near Count Rates, Average, Computed DH 1/s
- Neutron Monitor Activity Factor, Computed DH %

Tank and Volume Data:
- Tank volume (active) m3

## FUTURE PARSER IMPROVEMENTS

### 1. Catch-All Parser Strategy
**Problem**: Currently, files that don't match any configured extension are ignored
**Proposed Solution**: Use UnstructuredParser as a catch-all parser for unknown file types

**Implementation Ideas**:
- Add a "catch-all" mode to UnstructuredParser that accepts any file type
- Modify file classification logic to route unmatched files to UnstructuredParser
- Create a fallback chain: Structured → Unstructured → Ignore

**Benefits**:
- No data loss - all files get processed somehow
- Leverages Unstructured SDK's broad file type support
- Provides vector searchable content for unknown file types

### 2. Structured Parser Fallback Strategy
**Problem**: If structured parsers fail, files are completely lost
**Proposed Solution**: Implement fallback to UnstructuredParser when structured parsing fails

**Implementation Ideas**:
- Add error handling in structured parsers to catch parsing failures
- Implement retry logic: Structured Parser → Unstructured Parser
- Track which files were processed via fallback vs primary method

**Benefits**:
- Higher success rate for file processing
- Graceful degradation when structured parsing fails
- Maintains data accessibility even with parsing errors

### 3. Enhanced File Classification
**Current**: Binary classification (structured vs unstructured vs ignored)
**Proposed**: Multi-tier classification with fallbacks

**Classification Tiers**:
1. **Primary Structured**: LAS, DLIS, CSV with dedicated parsers
2. **Secondary Structured**: Other structured formats (JSON, XML, etc.)
3. **Primary Unstructured**: PDF, DOC, XLSX with UnstructuredParser
4. **Catch-All Unstructured**: Unknown files via UnstructuredParser
5. **Ignored**: Only truly unprocessable files (corrupted, system files)

### 4. Configuration-Driven Fallbacks
**Proposed .env Configuration**:
```bash
# Primary structured parsers
STRUCTURED_FILE_TYPES=.las,.dlis,.csv

# Primary unstructured parsers  
UNSTRUCTURED_FILE_TYPES=.pdf,.docx,.xlsx,.txt

# Fallback behavior
ENABLE_CATCH_ALL_PARSER=true
ENABLE_STRUCTURED_FALLBACK=true
MAX_FALLBACK_ATTEMPTS=2
```

### 5. Processing Pipeline Improvements
**Current Flow**:
```
File → Classification → Single Parser → Output
```

**Proposed Flow**:
```
File → Classification → Primary Parser → Success?
                                    ↓ No
                              Fallback Parser → Success?
                                              ↓ No
                                        Log & Ignore
```

### 6. Monitoring and Analytics
**Track Processing Methods**:
- Primary structured parsing success rate
- Fallback unstructured parsing success rate
- File types that frequently need fallbacks
- Performance impact of fallback processing

**Benefits**:
- Identify which file types need dedicated parsers
- Optimize processing pipeline based on actual usage
- Improve success rates over time

### 7. Implementation Priority
1. **Phase 1**: Add catch-all UnstructuredParser for unmatched files
2. **Phase 2**: Implement structured parser fallback to unstructured
3. **Phase 3**: Enhanced configuration and monitoring
4. **Phase 4**: Performance optimization and analytics

### 8. Considerations
- **Performance**: Fallback processing adds overhead
- **Quality**: Unstructured parsing may be less precise than structured
- **Storage**: More files processed = more storage requirements
- **Complexity**: More complex error handling and retry logic