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
gamma_ray, sonic_transit_time
Various metadata and parameter information