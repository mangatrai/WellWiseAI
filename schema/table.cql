CREATE TABLE default_keyspace.petro_data (
    well_id text,
    record_type text,
    curve_name text,
    depth_start double,
    acquisition_date timestamp,
    analyst text,
    archie_a double,
    archie_m double,
    archie_n double,
    bulk_vol_water double,
    caliper double,
    carbonate_flag boolean,
    clay_volume double,
    coal_flag boolean,
    core_permeability double,
    core_porosity double,
    depth_end double,
    facies_code text,
    facies_confidence double,
    file_checksum text,
    file_origin text,
    formation_press double,
    formation_temp double,
    gas_oil_ratio double,
    horizon_depth double,
    horizon_name text,
    mud_flow_rate double,
    mud_viscosity double,
    mud_weight_actual double,
    null_count int,
    num_samples int,
    permeability double,
    plan_azimuth double,
    plan_inclination double,
    plan_md double,
    plan_tvd double,
    porosity double,
    processing_date timestamp,
    processing_software text,
    production_rate double,
    pump_pressure double,
    qc_flag boolean,
    remarks text,
    resistivity_deep double,
    resistivity_medium double,
    resistivity_shallow double,
    rop double,
    sample_interval double,
    sample_max double,
    sample_mean double,
    sample_min double,
    sample_stddev double,
    sand_flag boolean,
    seismic_inline int,
    seismic_sample_rate double,
    seismic_trace_count int,
    seismic_xline int,
    service_company text,
    sp_curves double,
    tool_type text,
    torque double,
    version text,
    vp double,
    vs double,
    vshale double,
    water_resistivity double,
    water_saturation double,
    weight_on_bit double,
    PRIMARY KEY ((well_id, record_type), curve_name, depth_start)
) WITH CLUSTERING ORDER BY (curve_name ASC, depth_start ASC)
    AND additional_write_policy = '99p'
    AND bloom_filter_fp_chance = 0.01
    AND caching = {'keys': 'ALL', 'rows_per_partition': 'NONE'}
    AND comment = 'Wide, single-table schema (75 cols) for all structured petro data—partitioned by well & record type, clustered by curve & depth for performant depth-range and curve-specific queries'
    AND compaction = {'class': 'org.apache.cassandra.db.compaction.UnifiedCompactionStrategy'}
    AND compression = {'chunk_length_in_kb': '16', 'class': 'org.apache.cassandra.io.compress.LZ4Compressor'}
    AND crc_check_chance = 1.0
    AND default_time_to_live = 0
    AND gc_grace_seconds = 864000
    AND max_index_interval = 2048
    AND memtable_flush_period_in_ms = 0
    AND min_index_interval = 128
    AND read_repair = 'BLOCKING'
    AND speculative_retry = '99p';
CREATE CUSTOM INDEX petro_data_facies_code_idx ON default_keyspace.petro_data (facies_code) USING 'StorageAttachedIndex';
CREATE CUSTOM INDEX petro_data_gas_oil_ratio_idx ON default_keyspace.petro_data (gas_oil_ratio) USING 'StorageAttachedIndex';
CREATE CUSTOM INDEX petro_data_horizon_name_idx ON default_keyspace.petro_data (horizon_name) USING 'StorageAttachedIndex';
CREATE CUSTOM INDEX petro_data_production_rate_idx ON default_keyspace.petro_data (production_rate) USING 'StorageAttachedIndex';
CREATE CUSTOM INDEX petro_data_service_company_idx ON default_keyspace.petro_data (service_company) USING 'StorageAttachedIndex';
CREATE CUSTOM INDEX petro_data_tool_type_idx ON default_keyspace.petro_data (tool_type) USING 'StorageAttachedIndex';