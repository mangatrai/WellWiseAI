#!/usr/bin/env python3
"""
Astra DB Table Management Utility
Clean and manage tables in Astra DB
"""

import os
import argparse
import logging
from astrapy import DataAPIClient
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
    keyspace = os.getenv('ASTRA_DB_KEYSPACE')
    
    if not api_endpoint or not token:
        raise ValueError("Missing Astra DB credentials. Set ASTRA_DB_API_ENDPOINT and ASTRA_DB_TOKEN in .env file")
    
    # Initialize Astra DB client
    client = DataAPIClient()
    
    # Connect to database
    if keyspace:
        database = client.get_database(api_endpoint, token=token, keyspace=keyspace)
        print(f"🔗 Connected to Astra DB with keyspace: {keyspace}")
    else:
        database = client.get_database(api_endpoint, token=token)
        print(f"🔗 Connected to Astra DB (default keyspace)")
    
    return database

def truncate_table(database, table_name: str):
    """Truncate a table by deleting all records"""
    try:
        print(f"🗑️  Truncating table: {table_name}")
        
        # Get the table
        table = database.get_table(table_name)
        
        # Delete all records with empty filter (truncate)
        table.delete_many({})
        
        print(f"✅ Successfully truncated table: {table_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error truncating table {table_name}: {e}")
        return False

def drop_table(database, table_name: str):
    """Drop a table completely"""
    try:
        print(f"🗑️  Dropping table: {table_name}")
        
        # Drop the table
        database.drop_table(table_name, if_exists=True)
        
        print(f"✅ Successfully dropped table: {table_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error dropping table {table_name}: {e}")
        return False

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
    parser = argparse.ArgumentParser(description="Astra DB Table Management Utility")
    parser.add_argument("--action", choices=["truncate", "drop", "list"], required=True,
                       help="Action to perform: truncate, drop, or list tables")
    parser.add_argument("--table", help="Table name (optional, uses ASTRA_DB_TABLE from .env if not provided)")
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
            
        elif args.action in ["truncate", "drop"]:
            # Get table name
            table_name = args.table or os.getenv('ASTRA_TABLE_NAME')
            
            if not table_name:
                print("❌ Error: Table name required. Provide --table argument or set ASTRA_TABLE_NAME in .env file")
                return
            
            print(f"🎯 Target table: {table_name}")
            
            # Confirm action for destructive operations
            if args.action == "truncate":
                confirm = input(f"⚠️  Are you sure you want to TRUNCATE table '{table_name}'? (yes/no): ")
                if confirm.lower() != 'yes':
                    print("❌ Operation cancelled")
                    return
                success = truncate_table(database, table_name)
                
            elif args.action == "drop":
                confirm = input(f"⚠️  Are you sure you want to DROP table '{table_name}'? This cannot be undone! (yes/no): ")
                if confirm.lower() != 'yes':
                    print("❌ Operation cancelled")
                    return
                success = drop_table(database, table_name)
        
        if success:
            print("🎉 Operation completed successfully!")
        else:
            print("💥 Operation failed!")
            
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n📋 Setup Instructions:")
        print("1. Create a .env file in the project root")
        print("2. Add your Astra DB credentials:")
        print("   ASTRA_DB_API_ENDPOINT=your_api_endpoint_here")
        print("   ASTRA_DB_TOKEN=your_application_token_here")
        print("   ASTRA_DB_TABLE=your_table_name_here (optional)")
        print("3. Or pass credentials as command line arguments")
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        print(f"💥 Unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
