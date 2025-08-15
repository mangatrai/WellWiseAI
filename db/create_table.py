#!/usr/bin/env python3
"""
Astra DB Table Creation Utility
Create the petro_data table with all columns, primary key, and indexes
"""

import os
import argparse
import logging
from astrapy import DataAPIClient
from astrapy.constants import SortMode
from astrapy.info import (
    CreateTableDefinition,
    ColumnType,
    TableScalarColumnTypeDescriptor,
    TablePrimaryKeyDescriptor,
    TableIndexOptions
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_database_connection():
    """Get Astra DB database connection"""
    # Get credentials from environment variables
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    token = os.getenv('ASTRA_DB_TOKEN')
    keyspace = os.getenv('ASTRA_DB_KEYSPACE', 'default_keyspace')
    
    if not api_endpoint or not token:
        raise ValueError("Missing Astra DB credentials. Set ASTRA_DB_API_ENDPOINT and ASTRA_DB_TOKEN in .env file")
    
    # Initialize Astra DB client
    client = DataAPIClient()
    
    # Connect to database
    database = client.get_database(api_endpoint, token=token, keyspace=keyspace)
    print(f"🔗 Connected to Astra DB with keyspace: {keyspace}")
    
    return database

def create_petro_data_table_definition():
    """Create the table definition for petro_data table"""
    
    # Define all columns with their data types
    columns = {
        # Primary key columns
        "well_id": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "record_type": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "curve_name": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "depth_start": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        
        # Timestamp columns
        "acquisition_date": TableScalarColumnTypeDescriptor(column_type=ColumnType.TIMESTAMP),
        "processing_date": TableScalarColumnTypeDescriptor(column_type=ColumnType.TIMESTAMP),
        
        # Text columns
        "analyst": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "file_checksum": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "file_origin": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "horizon_name": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "processing_software": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "service_company": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "tool_type": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "version": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "remarks": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "country": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "state_province": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "field_name": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "block_name": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        "facies_code": TableScalarColumnTypeDescriptor(column_type=ColumnType.TEXT),
        
        # Boolean columns
        "carbonate_flag": TableScalarColumnTypeDescriptor(column_type=ColumnType.BOOLEAN),
        "coal_flag": TableScalarColumnTypeDescriptor(column_type=ColumnType.BOOLEAN),
        "qc_flag": TableScalarColumnTypeDescriptor(column_type=ColumnType.BOOLEAN),
        "sand_flag": TableScalarColumnTypeDescriptor(column_type=ColumnType.BOOLEAN),
        
        # Integer columns
        "null_count": TableScalarColumnTypeDescriptor(column_type=ColumnType.INT),
        "num_samples": TableScalarColumnTypeDescriptor(column_type=ColumnType.INT),
        "seismic_inline": TableScalarColumnTypeDescriptor(column_type=ColumnType.INT),
        "seismic_trace_count": TableScalarColumnTypeDescriptor(column_type=ColumnType.INT),
        "seismic_xline": TableScalarColumnTypeDescriptor(column_type=ColumnType.INT),
        
        # Double columns (all remaining numeric fields)
        "archie_a": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "archie_m": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "archie_n": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "bulk_vol_water": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "caliper": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "clay_volume": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "core_permeability": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "core_porosity": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "depth_end": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "facies_confidence": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "formation_press": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "formation_temp": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "gas_oil_ratio": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "horizon_depth": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "mud_flow_rate": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "mud_viscosity": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "mud_weight_actual": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "permeability": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "plan_azimuth": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "plan_inclination": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "plan_md": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "plan_tvd": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "porosity": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "production_rate": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "pump_pressure": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "resistivity_deep": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "resistivity_medium": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "resistivity_shallow": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "rop": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "sample_interval": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "sample_max": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "sample_mean": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "sample_min": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "sample_stddev": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "seismic_sample_rate": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "sp_curves": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "torque": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "vp": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "vs": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "vshale": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "water_resistivity": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "water_saturation": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "weight_on_bit": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "latitude": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "longitude": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
        "elevation": TableScalarColumnTypeDescriptor(column_type=ColumnType.DOUBLE),
    }
    
    # Define the primary key: ((well_id, record_type), curve_name, depth_start)
    primary_key = TablePrimaryKeyDescriptor(
        partition_by=["well_id", "record_type"],
        partition_sort={
            "curve_name": SortMode.ASCENDING,
            "depth_start": SortMode.ASCENDING,
        },
    )
    
    # Create table definition
    table_definition = CreateTableDefinition(
        columns=columns,
        primary_key=primary_key,
    )
    
    return table_definition

def create_table(database, table_name: str, table_definition: CreateTableDefinition):
    """Create the table in Astra DB"""
    try:
        print(f"🏗️  Creating table: {table_name}")
        
        # Create the table
        table = database.create_table(
            table_name,
            definition=table_definition,
            if_not_exists=True,  # Don't fail if table already exists
        )
        
        print(f"✅ Successfully created table: {table_name}")
        return table
        
    except Exception as e:
        print(f"❌ Error creating table {table_name}: {e}")
        return None

def create_indexes(table, table_name: str):
    """Create all indexes for the petro_data table"""
    
    # Define indexes to create (matching the CQL schema)
    indexes = [
        ("petro_data_facies_code_idx", "facies_code"),
        ("petro_data_gas_oil_ratio_idx", "gas_oil_ratio"),
        ("petro_data_horizon_name_idx", "horizon_name"),
        ("petro_data_production_rate_idx", "production_rate"),
        ("petro_data_service_company_idx", "service_company"),
        ("petro_data_tool_type_idx", "tool_type"),
        ("petro_data_country_idx", "country"),
        ("petro_data_field_name_idx", "field_name"),
        ("petro_data_latitude_idx", "latitude"),
        ("petro_data_longitude_idx", "longitude"),
    ]
    
    created_count = 0
    failed_count = 0
    
    for index_name, column_name in indexes:
        try:
            print(f"📊 Creating index: {index_name} on column: {column_name}")
            
            # Create index with case-insensitive option for text columns
            if column_name in ["facies_code", "horizon_name", "service_company", "tool_type", "country", "field_name"]:
                # Text columns - use case-insensitive index
                table.create_index(
                    index_name,
                    column=column_name,
                    options=TableIndexOptions(case_sensitive=False),
                    if_not_exists=True,
                )
            else:
                # Numeric columns - use default options
                table.create_index(
                    index_name,
                    column=column_name,
                    if_not_exists=True,
                )
            
            print(f"✅ Successfully created index: {index_name}")
            created_count += 1
            
        except Exception as e:
            print(f"❌ Error creating index {index_name}: {e}")
            failed_count += 1
    
    print(f"\n📈 Index creation summary:")
    print(f"   ✅ Created: {created_count} indexes")
    print(f"   ❌ Failed: {failed_count} indexes")
    
    return created_count, failed_count

def list_tables(database):
    """List all tables in the database"""
    try:
        print("📋 Listing all tables:")
        
        # Get all table names
        tables = database.list_table_names()
        
        if tables:
            for i, table_name in enumerate(tables, 1):
                print(f"   {i}. {table_name}")
        else:
            print("   No tables found")
            
        return True
        
    except Exception as e:
        print(f"❌ Error listing tables: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Astra DB Table Creation Utility")
    parser.add_argument("--action", choices=["create", "list"], default="create",
                       help="Action to perform: create table or list tables")
    parser.add_argument("--table", default="petro_data",
                       help="Table name (default: petro_data)")
    parser.add_argument("--skip-indexes", action="store_true",
                       help="Skip creating indexes (only create table)")
    parser.add_argument("--api-endpoint", help="Astra DB API endpoint (or set ASTRA_DB_API_ENDPOINT in .env)")
    parser.add_argument("--token", help="Astra DB token (or set ASTRA_DB_TOKEN in .env)")
    parser.add_argument("--keyspace", help="Astra DB keyspace (or set ASTRA_DB_KEYSPACE in .env)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Override environment variables if provided
        if args.api_endpoint:
            os.environ['ASTRA_DB_API_ENDPOINT'] = args.api_endpoint
        if args.token:
            os.environ['ASTRA_DB_TOKEN'] = args.token
        if args.keyspace:
            os.environ['ASTRA_DB_KEYSPACE'] = args.keyspace
        
        # Get database connection
        database = get_database_connection()
        
        # Perform the requested action
        if args.action == "list":
            success = list_tables(database)
            
        elif args.action == "create":
            print(f"🎯 Creating table: {args.table}")
            
            # Create table definition
            table_definition = create_petro_data_table_definition()
            
            # Create the table
            table = create_table(database, args.table, table_definition)
            
            if table and not args.skip_indexes:
                print(f"\n🔗 Creating indexes for table: {args.table}")
                created_count, failed_count = create_indexes(table, args.table)
                
                if failed_count == 0:
                    print(f"🎉 Table '{args.table}' created successfully with all indexes!")
                else:
                    print(f"⚠️  Table '{args.table}' created but some indexes failed!")
            elif table:
                print(f"🎉 Table '{args.table}' created successfully (indexes skipped)!")
            else:
                print(f"💥 Failed to create table '{args.table}'!")
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n📋 Setup Instructions:")
        print("1. Create a .env file in the project root")
        print("2. Add your Astra DB credentials:")
        print("   ASTRA_DB_API_ENDPOINT=your_api_endpoint_here")
        print("   ASTRA_DB_TOKEN=your_application_token_here")
        print("   ASTRA_DB_KEYSPACE=your_keyspace_here (optional, defaults to 'default_keyspace')")
        print("3. Or pass credentials as command line arguments")
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        print(f"💥 Unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
