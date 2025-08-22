#!/usr/bin/env python3
"""
Astra DB Vector Store - Single Collection Approach
Simplified and more intelligent approach using one collection with rich metadata
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import glob

from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.info import (
    CollectionDefinition,
    CollectionVectorOptions,
    VectorServiceOptions,
    CollectionLexicalOptions,
    CollectionRerankOptions,
    RerankServiceOptions,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class VectorSearchResult:
    """Result from vector search"""
    content: str
    metadata: Dict[str, Any]
    document_type: str
    source: str
    similarity_score: float
    rerank_score: float
    vector_score: float
    id: str

class AstraVectorStoreSingle:
    """Astra DB Vector Store using a single collection with rich metadata"""
    
    def __init__(self, api_endpoint: str = None, token: str = None, keyspace: str = None):
        # Use environment variables if not provided
        self.api_endpoint = api_endpoint or os.getenv('ASTRA_DB_API_ENDPOINT')
        self.token = token or os.getenv('ASTRA_DB_TOKEN')
        self.keyspace = keyspace or os.getenv('ASTRA_DB_KEYSPACE')  # Let Astra DB use default
        
        # Validate required credentials
        if not self.api_endpoint or not self.token:
            raise ValueError("Missing Astra DB credentials. Set ASTRA_DB_API_ENDPOINT and ASTRA_DB_TOKEN in .env file")
        
        # Initialize logging
        def setup_vector_logging():
            """Setup logging for vector store operations."""
            log_level = os.getenv('LOG_LEVEL', 'INFO')
            
            # Convert string log level to logging constant
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR
            }
            log_level_constant = level_map.get(log_level.upper(), logging.INFO)
            
            # Get logger for vector store operations
            logger = logging.getLogger(self.__class__.__name__)
            logger.setLevel(log_level_constant)
            
            # Only add handlers if they don't already exist (avoid duplicates)
            if not logger.handlers:
                # Console handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(log_level_constant)
                
                # File handler - use environment variable
                log_file = os.getenv('LOG_FILE_NAME', 'wellwise_parser.log')
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)  # Always log everything to file
                
                # Formatter
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(formatter)
                file_handler.setFormatter(formatter)
                
                logger.addHandler(console_handler)
                logger.addHandler(file_handler)
            
            return logger
        
        self.logger = setup_vector_logging()
        
        # Single collection name - get from environment or use default
        self.collection_name = os.getenv('ASTRA_DB_COLLECTION', 'oil_gas_documents')

        # Embedding model name
        self.embded_model = os.getenv('EMBED_MODEL', 'text-embedding-ada-002')

        #astra api key name
        self.astra_api_key_name = os.getenv('ASTRA_OPENAI_KEY_NAME', 'openai-key')
        
        # Initialize Astra DB client
        self.client = DataAPIClient()
        print(f"🔗 Connecting to Astra DB:")
        print(f"   API Endpoint: {self.api_endpoint}")
        
        # Connect to database with or without explicit keyspace
        if self.keyspace:
            print(f"   Keyspace: {self.keyspace}")
            self.database = self.client.get_database(
                self.api_endpoint,
                token=self.token,
                keyspace=self.keyspace
            )
        else:
            print(f"   Keyspace: (using default)")
            self.database = self.client.get_database(
                self.api_endpoint,
                token=self.token
            )
        
        # Initialize collection
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize the single collection in Astra DB with hybrid search support"""
        print("🔧 Initializing Astra DB collection with hybrid search support...")
        
        try:
            # Check if collection exists using list_collection_names
            print(f"  📋 Checking if collection '{self.collection_name}' exists...")
            existing_collections = self.database.list_collection_names()
            
            if self.collection_name in existing_collections:
                print(f"  ✅ Collection '{self.collection_name}' already exists")
            else:
                print(f"  📝 Collection '{self.collection_name}' doesn't exist, creating...")
                # Collection doesn't exist, create it with hybrid search support
                collection_definition = CollectionDefinition(
                    vector=CollectionVectorOptions(
                        metric=VectorMetric.COSINE,
                        dimension=1536,
                        service=VectorServiceOptions(
                            provider="openai",
                            model_name=self.embded_model,
                            authentication={
                                "providerKey": self.astra_api_key_name,
                            },
                        )
                    ),
                    lexical=CollectionLexicalOptions(
                        analyzer={
                            "tokenizer": {"name": "standard", "args": {}},
                            "filters": [
                                {"name": "lowercase"},
                                {"name": "stop"},
                                {"name": "porterstem"},
                                {"name": "asciifolding"},
                            ],
                            "charFilters": [],
                        },
                        enabled=True,
                    ),
                    rerank=CollectionRerankOptions(
                        enabled=True,
                        service=RerankServiceOptions(
                            provider="nvidia",
                            model_name="nvidia/llama-3.2-nv-rerankqa-1b-v2",
                        ),
                    ),
                )
                
                print(f"  🏗️ Creating collection '{self.collection_name}' with hybrid search support...")
                collection = self.database.create_collection(
                    self.collection_name,
                    definition=collection_definition,
                )
                print(f"  ✅ Created collection '{self.collection_name}' with hybrid search support")
                
        except Exception as e:
            print(f"  ❌ Error initializing collection '{self.collection_name}': {e}")
            import traceback
            print(f"  📋 Full traceback: {traceback.format_exc()}")
            raise
    
    def load_contextual_documents(self, documents_file: str = None, documents_dir: str = None):
        """Load contextual documents into the single collection
        
        Automatically finds the most recent contextual documents file if not specified
        """
        if documents_dir:
            # Load all unstructured files from directory
            pattern = os.path.join(documents_dir, "*_unstructured.json")
            files = glob.glob(pattern)
            if not files:
                self.logger.error(f"❌ No *_unstructured.json files found in directory: {documents_dir}")
                return False
            
            print(f"📁 Found {len(files)} unstructured files in directory: {documents_dir}")
            total_loaded = 0
            
            for file_path in files:
                print(f"📄 Processing: {os.path.basename(file_path)}")
                success = self._load_single_file(file_path)
                if success:
                    total_loaded += 1
            
            print(f"🎉 Successfully processed {total_loaded}/{len(files)} files")
            return total_loaded > 0
            
        elif documents_file:
            # Load single specified file
            return self._load_single_file(documents_file)
            
        else:
            # Auto-detect: Look for *_unstructured.json in UNST_PARSED_DATA_DIR
            default_dir = os.getenv('UNST_PARSED_DATA_DIR', 'unstructured_data')
            pattern = os.path.join(default_dir, "*_unstructured.json")
            files = glob.glob(pattern)
            if not files:
                self.logger.error(f"❌ No *_unstructured.json files found in default directory: {default_dir}")
                return False
            
            print(f"📁 Found {len(files)} unstructured files in default directory: {default_dir}")
            total_loaded = 0
            
            for file_path in files:
                print(f"📄 Processing: {os.path.basename(file_path)}")
                success = self._load_single_file(file_path)
                if success:
                    total_loaded += 1
            
            print(f"🎉 Successfully processed {total_loaded}/{len(files)} files")
            return total_loaded > 0
    
    def _load_single_file(self, documents_file: str) -> bool:
        """Load a single JSON file into the collection"""
        if not os.path.exists(documents_file):
            self.logger.error(f"❌ Documents file not found: {documents_file}")
            return False
        
        try:
            with open(documents_file, 'r') as f:
                documents = json.load(f)
            
            print(f"📄 Found {len(documents)} documents to load from {os.path.basename(documents_file)}")
            
            # Load all documents into the single collection
            loaded_count = self._load_documents_to_collection(documents)
            
            print(f"🎉 Successfully loaded {loaded_count} documents to Astra DB from {os.path.basename(documents_file)}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading documents from {os.path.basename(documents_file)}: {e}")
            import traceback
            print(f"📋 Full traceback: {traceback.format_exc()}")
            return False
    
    def _load_documents_to_collection(self, documents: List[Dict]) -> int:
        """Load documents into the single collection using Astra's hybrid search service"""
        try:
            collection = self.database.get_collection(self.collection_name)
            
            # Handle new structure with contextual_documents key
            if isinstance(documents, dict) and 'contextual_documents' in documents:
                documents = documents['contextual_documents']
            
            # Prepare documents for insertion with direct field access
            documents_to_insert = []
            for doc in documents:
                # Generate a unique ID
                doc_id = str(uuid.uuid4())
                
                # Create minimal metadata with only essential fields
                metadata = {
                    'created_at': datetime.now().isoformat(),
                    'chunk_id': doc.get('metadata', {}).get('chunk_id', 0),
                    'element_type': doc.get('metadata', {}).get('element_type', ''),
                    'total_chunks': doc.get('metadata', {}).get('total_chunks', 0)
                }
                
                # Simple hybrid content: well_id + content
                well_id = doc.get('well_id', '')
                hybrid_content = f"{well_id} {doc['content']}" if well_id else doc['content']
                
                # Create document for Astra DB with direct field access
                astra_doc = {
                    "_id": doc_id,
                    "content": doc['content'],
                    "well_id": doc.get('well_id', ''),
                    "depth_references": doc.get('depth_references', []),
                    "technical_terms": doc.get('technical_terms', []),
                    "mathematical_formulas": doc.get('mathematical_formulas', []),
                    "geological_context": doc.get('geological_context', {}),
                    "curve_names": doc.get('curve_names', []),
                    "document_type": doc.get('document_type', ''),
                    "source": doc.get('source', ''),
                    "timestamp": doc.get('timestamp', ''),
                    "metadata": metadata,
                    "$hybrid": hybrid_content
                }
                
                documents_to_insert.append(astra_doc)
            
            # Insert documents in batches
            batch_size = 10
            loaded_count = 0
            
            for i in range(0, len(documents_to_insert), batch_size):
                batch = documents_to_insert[i:i + batch_size]
                try:
                    result = collection.insert_many(batch)
                    loaded_count += len(batch)
                    print(f"  📝 Inserted batch {i//batch_size + 1} ({len(batch)} documents)")
                except Exception as e:
                    self.logger.error(f"❌ Error inserting batch: {e}")
            
            return loaded_count
            
        except Exception as e:
            self.logger.error(f"❌ Error loading to collection {self.collection_name}: {e}")
            import traceback
            print(f"📋 Full traceback: {traceback.format_exc()}")
            return 0
    

    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for the collection"""
        try:
            collection = self.database.get_collection(self.collection_name)
            count = collection.estimated_document_count()
            
            # Get document type distribution manually since aggregate might not be available
            document_types = {}
            try:
                # Try to get all documents and count by type
                all_docs = collection.find({}, limit=1000)
                for doc in all_docs:
                    doc_type = doc.get('document_type', 'unknown')
                    document_types[doc_type] = document_types.get(doc_type, 0) + 1
            except Exception as e:
                self.logger.warning(f"Could not get document type distribution: {e}")
            
            return {
                'total_documents': count,
                'document_types': document_types
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting stats: {e}")
            return {'total_documents': 0, 'document_types': {}}
    
    def search_documents(self, query: str, 
                        document_types: List[str] = None,
                        well_ids: List[str] = None,
                        technical_terms: List[str] = None,
                        formations: List[str] = None,
                        limit: int = 5, 
                        similarity_threshold: float = 0.7) -> List[VectorSearchResult]:
        """Search documents using vector similarity with optional filtering"""
        try:
            collection = self.database.get_collection(self.collection_name)
            
            # Build filter based on direct fields
            filter_query = {}
            
            if document_types:
                filter_query["document_type"] = {"$in": document_types}
            
            if well_ids:
                filter_query["well_id"] = {"$in": well_ids}
            
            if technical_terms:
                filter_query["technical_terms"] = {"$in": technical_terms}
            
            if formations:
                filter_query["geological_context.formations"] = {"$in": formations}
            
            # Use vectorize service for semantic search
            search_results = collection.find(
                filter_query,  # Apply metadata filters
                sort={"$vectorize": query},  # Sort by vector similarity
                limit=limit,
                include_similarity=True  # Include similarity scores
            )
            
            results = []
            for result in search_results:
                search_result = VectorSearchResult(
                    content=result.get('content', ''),
                    metadata=result.get('metadata', {}),
                    document_type=result.get('document_type', ''),
                    source=result.get('source', ''),
                    similarity_score=result.get('$similarity', 0.8),
                    id=result.get('_id', '')
                )
                results.append(search_result)
            
            # Sort by similarity score and apply threshold
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            results = [r for r in results if r.similarity_score >= similarity_threshold]
            
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"❌ Error in document search: {e}")
            return []
    
    def search_documents_hybrid(self, query: str, 
                               document_types: List[str] = None,
                               well_ids: List[str] = None,
                               technical_terms: List[str] = None,
                               formations: List[str] = None,
                               limit: int = 5) -> List[VectorSearchResult]:
        """Search documents using hybrid search (vector + lexical + reranking)"""
        try:
            collection = self.database.get_collection(self.collection_name)
            
            # Build filter based on direct fields
            filter_query = {}
            
            if document_types:
                filter_query["document_type"] = {"$in": document_types}
            
            if well_ids:
                filter_query["well_id"] = {"$in": well_ids}
            
            if technical_terms:
                filter_query["technical_terms"] = {"$in": technical_terms}
            
            if formations:
                filter_query["geological_context.formations"] = {"$in": formations}
            
            # Use hybrid search with reranking
            search_results = collection.find_and_rerank(
                filter_query,  # Apply metadata filters
                sort={"$hybrid": query},  # Hybrid search
                limit=limit,
                include_scores=True
            )
            
            results = []
            for result in search_results:
                # Get scores from the result object
                scores = result.scores
                rerank_score = scores.get('$rerank', 0)
                vector_score = scores.get('$vector', 0)
                
                # Get document data from result.document
                document = result.document
                search_result = VectorSearchResult(
                    content=document.get('content', ''),
                    metadata=document.get('metadata', {}),
                    document_type=document.get('document_type', ''),
                    source=document.get('source', ''),
                    similarity_score=rerank_score,  # Use rerank score as primary similarity
                    rerank_score=rerank_score,
                    vector_score=vector_score,
                    id=document.get('_id', '')
                )
                results.append(search_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in hybrid search: {e}")
            return []
    
    def get_documents_by_well(self, well_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all documents related to a specific well"""
        try:
            collection = self.database.get_collection(self.collection_name)
            
            results = collection.find(
                {"well_id": well_id},
                limit=limit
            )
            return list(results)
            
        except Exception as e:
            self.logger.error(f"❌ Error getting documents by well: {e}")
            return []
    
    def get_documents_by_type(self, document_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get documents by type"""
        try:
            collection = self.database.get_collection(self.collection_name)
            
            results = collection.find(
                {"document_type": document_type},
                limit=limit
            )
            return list(results)
            
        except Exception as e:
            self.logger.error(f"❌ Error getting documents by type: {e}")
            return []

class AstraQueryEngineSingle:
    """Enhanced query engine for single collection approach"""
    
    def __init__(self, vector_store: AstraVectorStoreSingle):
        self.vector_store = vector_store
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def query(self, question: str, limit: int = 5) -> List[VectorSearchResult]:
        """Query the vector store with hybrid search"""
        print(f"🔍 Querying with hybrid search: {question}")
        
        # Analyze the question to determine what data types might be relevant
        relevant_types = self._analyze_query_requirements(question)
        
        # Use hybrid search with relevant document type filtering
        results = self.vector_store.search_documents_hybrid(
            query=question,
            document_types=relevant_types,
            limit=limit
        )
        
        # If no results with filtering, try broader search
        if not results:
            print(f"  🔄 No results with filtered search, trying broader search...")
            results = self.vector_store.search_documents_hybrid(
                query=question,
                limit=limit
            )
        
        return results
    
    def _analyze_query_requirements(self, question: str) -> List[str]:
        """Analyze question to determine relevant document types"""
        question_lower = question.lower()
        relevant_types = []
        
        # Well-related queries
        if any(word in question_lower for word in ['well', 'wells', 'borehole']):
            relevant_types.extend(['unstructured_compositeelement', 'dlis', 'segy'])
        
        # Petrophysical queries
        if any(word in question_lower for word in ['petrophysical', 'permeability', 'porosity', 'log', 'logs', 'las', 'asc']):
            relevant_types.extend(['unstructured_compositeelement', 'dlis'])
        
        # Seismic queries
        if any(word in question_lower for word in ['seismic', 'seism', 'migration', 'geophysics']):
            relevant_types.extend(['unstructured_compositeelement', 'segy'])
        
        # Formation/geological queries
        if any(word in question_lower for word in ['formation', 'geological', 'geology', 'stratigraphy', 'facies']):
            relevant_types.extend(['unstructured_compositeelement', 'unstructured_tablechunk'])
        
        # Table/data queries
        if any(word in question_lower for word in ['table', 'data', 'spreadsheet', 'excel']):
            relevant_types.append('unstructured_tablechunk')
        
        # If no specific types identified, return all
        if not relevant_types:
            relevant_types = ['unstructured_compositeelement', 'unstructured_tablechunk', 'dlis', 'segy']
        
        return list(set(relevant_types))  # Remove duplicates
    
    def query_well_comprehensive(self, well_name: str, limit: int = 10) -> List[VectorSearchResult]:
        """Get comprehensive information about a specific well using hybrid search"""
        print(f"🔍 Getting comprehensive data for well: {well_name}")
        
        # Search for all documents related to this well using hybrid search
        results = self.vector_store.search_documents_hybrid(
            query=f"well {well_name}",
            well_ids=[well_name],
            limit=limit
        )
        
        return results
    
    def query_cross_reference(self, question: str, well_name: str = None, limit: int = 5) -> List[VectorSearchResult]:
        """Query with cross-referencing capabilities using hybrid search"""
        print(f"🔍 Cross-reference query with hybrid search: {question}")
        
        # Build a more specific query
        if well_name:
            enhanced_query = f"{question} well {well_name}"
        else:
            enhanced_query = question
        
        # Search across all document types using hybrid search
        results = self.vector_store.search_documents_hybrid(
            query=enhanced_query,
            limit=limit
        )
        
        return results

def main():
    """Test the single collection Astra DB vector store"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Astra DB Vector Store - Single Collection")
    parser.add_argument("--api-endpoint", help="Astra DB API endpoint (or set ASTRA_DB_API_ENDPOINT in .env)")
    parser.add_argument("--token", help="Astra DB token (or set ASTRA_DB_TOKEN in .env)")
    parser.add_argument("--keyspace", help="Astra DB keyspace (or set ASTRA_DB_KEYSPACE in .env)")
    parser.add_argument("--load-documents", action="store_true", help="Load documents into Astra DB")
    parser.add_argument("--documents-dir", help="Directory containing unstructured JSON files")
    parser.add_argument("--documents-file", help="Specific JSON file to load")
    parser.add_argument("--query", help="Test query")
    parser.add_argument("--well", help="Specific well name for queries")
    
    args = parser.parse_args()
    
    # Initialize vector store
    print("🔧 Initializing Astra DB Vector Store (Single Collection)...")
    try:
        vector_store = AstraVectorStoreSingle(args.api_endpoint, args.token, args.keyspace)
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n📋 Setup Instructions:")
        print("1. Create a .env file in the project root")
        print("2. Add your Astra DB credentials:")
        print("   ASTRA_DB_API_ENDPOINT=your_api_endpoint_here")
        print("   ASTRA_DB_TOKEN=your_application_token_here")
        print("3. Or pass credentials as command line arguments")
        return
    
    if args.load_documents:
        # Load documents
        if args.documents_file:
            success = vector_store.load_contextual_documents(documents_file=args.documents_file)
        elif args.documents_dir:
            success = vector_store.load_contextual_documents(documents_dir=args.documents_dir)
        else:
            success = vector_store.load_contextual_documents()
        if success:
            # Show statistics
            stats = vector_store.get_collection_stats()
            print("\n📊 Collection Statistics:")
            print(f"   - Total documents: {stats.get('total_documents', 0)}")
            print("   - Document types:")
            for doc_type, count in stats.get('document_types', {}).items():
                print(f"     * {doc_type}: {count} documents")
    
    if args.query:
        # Test query
        query_engine = AstraQueryEngineSingle(vector_store)
        
        if args.well:
            # Well-specific query
            results = query_engine.query_cross_reference(args.query, args.well)
        else:
            # General query
            results = query_engine.query(args.query)
        
        print(f"\n🔍 Query Results for: '{args.query}'")
        print(f"   Found {len(results)} results")
        
        for i, result in enumerate(results, 1):
            print(f"\n   {i}. {result.document_type} (Rerank: {result.rerank_score:.2f}, Vector: {result.vector_score:.2f})")
            print(f"      Source: {result.source}")
            print(f"      Well: {result.metadata.get('well_name', 'N/A')}")
            print(f"      Content: {result.content[:200]}...")
    
    # Test vector search directly
    if args.load_documents:
        print(f"\n🧪 Testing hybrid search...")
        try:
            # Test a simple hybrid search
            test_results = vector_store.search_documents_hybrid("well data", limit=3)
            print(f"   Hybrid search test found {len(test_results)} results")
            if test_results:
                print(f"   First result rerank score: {test_results[0].rerank_score:.3f}")
        except Exception as e:
            print(f"   ❌ Hybrid search test failed: {e}")

if __name__ == "__main__":
    main() 