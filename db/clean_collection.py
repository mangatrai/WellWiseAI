#!/usr/bin/env python3
"""
Cleanup script to remove any specified Astra DB collection
"""

import os
import argparse
from astrapy import DataAPIClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def delete_collection(collection_name: str, keyspace: str = None):
    api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
    token = os.getenv('ASTRA_DB_TOKEN')
    env_keyspace = os.getenv('ASTRA_DB_KEYSPACE')

    if not api_endpoint or not token:
        print("❌ Missing Astra DB credentials in .env file")
        return

    if not collection_name:
        print("❌ Collection name must be provided.")
        return

    if not keyspace:
        if env_keyspace:
            keyspace = env_keyspace
            print(f"ℹ️  Using keyspace from .env: {keyspace}")
        else:
            print("❌ Keyspace must be provided via --keyspace or in .env file.")
            return

    print(f"🧹 Preparing to delete collection '{collection_name}' from keyspace '{keyspace}'...")
    print(f"   API Endpoint: {api_endpoint}")

    try:
        client = DataAPIClient()
        database = client.get_database(api_endpoint, token=token, keyspace=keyspace)
        try:
            collection = database.get_collection(collection_name)
            count = collection.estimated_document_count()
            print(f"   📋 Collection '{collection_name}' exists and has {count} documents.")
        except Exception as e:
            print(f"❌ Collection '{collection_name}' not found in keyspace '{keyspace}'.")
            return

        response = input(f"\n❓ Are you sure you want to delete collection '{collection_name}'? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("   ❌ Cleanup cancelled.")
            return

        try:
            database.drop_collection(collection_name)
            print(f"   ✅ Dropped collection: {collection_name}")
            print("\n🎉 Cleanup completed!")
        except Exception as e:
            print(f"❌ Error deleting collection '{collection_name}': {e}")
    except Exception as e:
        print(f"❌ Error connecting to Astra DB: {e}")

def main():
    parser = argparse.ArgumentParser(description="Delete any Astra DB collection by name.")
    parser.add_argument('--collection', required=True, help='Name of the collection to delete')
    parser.add_argument('--keyspace', help='Keyspace name (optional, will use .env if not provided)')
    args = parser.parse_args()
    delete_collection(args.collection, args.keyspace)

if __name__ == "__main__":
    main() 