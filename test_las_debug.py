import lasio
from parsers.las import extract_well_info, map_curve_to_canonical

# Load LAS file
las = lasio.read('/Users/mrai/Library/CloudStorage/Box-Box/Volve/PETROPHYSICAL INTERPRETATION/15_9-19 BT2/CPI/15_9-19_BT2_CPI.las')

# Extract well info
well_info = extract_well_info(las, None)
print('Well Info:')
print('well_id:', repr(well_info['well_id']))
print('start_depth:', repr(well_info['start_depth']))

# Test mapping
curve_data = {
    'curve_name': 'BVW',
    'mean': 0.123,
    'min': 0.0,
    'max': 0.5,
    'count': 100
}

file_info = {
    'file_path': '/test/path/file.las',
    'file_size': 1000,
    'file_type': 'LAS'
}

record = map_curve_to_canonical('BVW', curve_data, well_info, file_info, None)
print('\nMapped Record:')
print('well_id:', repr(record['well_id']))
print('depth_start:', repr(record['depth_start']))
print('bulk_volume_water:', repr(record['bulk_volume_water'])) 