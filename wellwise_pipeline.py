#!/usr/bin/env python3
"""
WellWiseAI Complete Pipeline: Parse and Insert Data
"""

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from wellwise_parser import main as parse_data
from db.insert_data import WellWiseDBInserter
from db.vector_store import AstraVectorStoreSingle

# Load environment variables
load_dotenv()

# Configure unified logging
def setup_unified_logging():
    """Setup unified logging for the entire pipeline."""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_file = os.getenv('LOG_FILE_NAME', 'wellwise_pipeline.log')
    
    # Convert string log level to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    log_level_constant = level_map.get(log_level.upper(), logging.INFO)
    
    # Get logger for pipeline
    logger = logging.getLogger('wellwise')
    logger.setLevel(log_level_constant)
    
    # Only add handlers if they don't already exist (avoid duplicates)
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level_constant)
        
        # File handler - use environment variable
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

# Setup unified logging
logger = setup_unified_logging()

class WellWisePipeline:
    """Complete pipeline for parsing and inserting oil and gas data."""
    
    def __init__(self):
        """
        Initialize the pipeline.
        """
        # Initialize inserter - it will handle its own credentials
        self.inserter = WellWiseDBInserter()
        self.db_available = self.inserter.is_available()
        
        # Initialize vector store for unstructured data
        try:
            self.vector_store = AstraVectorStoreSingle()
            self.vector_store_available = True
        except Exception as e:
            logger.warning(f"Vector store not available: {e}")
            self.vector_store = None
            self.vector_store_available = False
    
    def run_parsing(self) -> bool:
        """
        Run the data parsing step.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("🚀 Starting data parsing step...")
        
        try:
            # Run the parser
            parse_data()
            logger.info("✅ Data parsing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during data parsing: {e}")
            return False
    
    def run_database_insertion(self) -> dict:
        """
        Run the database insertion step.
        
        Returns:
            Dictionary with insertion results
        """
        if not self.db_available:
            logger.error("❌ Database not available. Cannot perform insertion.")
            return {'successful': 0, 'failed': 0, 'total': 0}
        
        logger.info("🚀 Starting database insertion step...")
        
        try:
            # Insert all data from parsed_data directory
            result = self.inserter.insert_from_parsed_data_directory(parsed_data_dir=os.getenv('PARSED_DIRECTORY', 'structured_data'))
            
            logger.info("✅ Database insertion completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during database insertion: {e}")
            return {'successful': 0, 'failed': 0, 'total': 0}
    
    def run_vector_store_insertion(self) -> dict:
        """
        Run the vector store insertion step for unstructured data.
        
        Returns:
            Dictionary with insertion results
        """
        if not self.vector_store_available:
            logger.error("❌ Vector store not available. Cannot perform unstructured data insertion.")
            return {'successful': 0, 'failed': 0, 'total': 0}
        
        logger.info("🚀 Starting vector store insertion step...")
        
        try:
            # Get unstructured data directory from environment
            unstructured_dir = os.getenv('UNST_PARSED_DATA_DIR', 'unstructured_data')
            
            # Load and insert unstructured documents
            success = self.vector_store.load_contextual_documents(documents_dir=unstructured_dir)
            
            logger.info("✅ Vector store insertion completed")
            
            # Return result in expected format
            if success:
                return {'successful': 1, 'failed': 0, 'total': 1}
            else:
                return {'successful': 0, 'failed': 1, 'total': 1}
            
        except Exception as e:
            logger.error(f"❌ Error during vector store insertion: {e}")
            return {'successful': 0, 'failed': 0, 'total': 0}
    
    def run_complete_pipeline(self, skip_parsing: bool = False, skip_insertion: bool = False) -> dict:
        """
        Run the complete pipeline: parsing + database insertion.
        
        Args:
            skip_parsing: Skip the parsing step if data already exists
            skip_insertion: Skip the database insertion step
            
        Returns:
            Dictionary with pipeline results
        """
        start_time = datetime.now()
        logger.info("🎯 Starting WellWiseAI Complete Pipeline")
        logger.info(f"⏰ Start time: {start_time}")
        
        results = {
            'parsing_success': False,
            'insertion_success': False,
            'vector_store_success': False,
            'parsing_records': 0,
            'insertion_results': {'successful': 0, 'failed': 0, 'total': 0},
            'vector_store_results': {'successful': 0, 'failed': 0, 'total': 0},
            'start_time': start_time,
            'end_time': None,
            'duration': None
        }
        
        # Step 1: Data Parsing
        if not skip_parsing:
            logger.info("=" * 50)
            logger.info("STEP 1: DATA PARSING")
            logger.info("=" * 50)
            
            parsing_success = self.run_parsing()
            results['parsing_success'] = parsing_success
            
            if not parsing_success:
                logger.error("❌ Pipeline failed at parsing step")
                return results
        else:
            logger.info("⏭️  Skipping parsing step (data already exists)")
            results['parsing_success'] = True
        
        # Step 2: Database Insertion (Structured Data)
        if not skip_insertion:
            logger.info("=" * 50)
            logger.info("STEP 2: STRUCTURED DATA INSERTION")
            logger.info("=" * 50)
            
            insertion_results = self.run_database_insertion()
            results['insertion_results'] = insertion_results
            results['insertion_success'] = insertion_results['successful'] > 0
            
            if insertion_results['successful'] == 0:
                logger.error("❌ Pipeline failed at structured data insertion step")
        else:
            logger.info("⏭️  Skipping structured data insertion step")
            results['insertion_success'] = True
        
        # Step 3: Vector Store Insertion (Unstructured Data)
        if not skip_insertion:
            logger.info("=" * 50)
            logger.info("STEP 3: UNSTRUCTURED DATA INSERTION")
            logger.info("=" * 50)
            
            vector_store_results = self.run_vector_store_insertion()
            results['vector_store_results'] = vector_store_results
            results['vector_store_success'] = vector_store_results['successful'] > 0
            
            if vector_store_results['successful'] == 0:
                logger.error("❌ Pipeline failed at unstructured data insertion step")
        else:
            logger.info("⏭️  Skipping unstructured data insertion step")
            results['vector_store_success'] = True
        
        # Calculate timing
        end_time = datetime.now()
        duration = end_time - start_time
        
        results['end_time'] = end_time
        results['duration'] = duration
        
        # Print summary
        self._print_pipeline_summary(results)
        
        return results
    
    def _print_pipeline_summary(self, results: dict):
        """Print a summary of the pipeline results."""
        logger.info("=" * 50)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 50)
        
        # Timing
        duration = results['duration']
        if duration:
            logger.info(f"⏱️  Total Duration: {duration}")
        
        # Parsing results
        if results['parsing_success']:
            logger.info("✅ Parsing: SUCCESS")
        else:
            logger.info("❌ Parsing: FAILED")
        
        # Structured data insertion results
        insertion_results = results['insertion_results']
        if results['insertion_success']:
            logger.info("✅ Structured Data Insertion: SUCCESS")
            logger.info(f"📊 Structured Data Results:")
            logger.info(f"   Total Records: {insertion_results['total']}")
            logger.info(f"   Successful: {insertion_results['successful']}")
            logger.info(f"   Failed: {insertion_results['failed']}")
            
            if insertion_results['total'] > 0:
                success_rate = (insertion_results['successful'] / insertion_results['total']) * 100
                logger.info(f"   Success Rate: {success_rate:.1f}%")
        else:
            logger.info("❌ Structured Data Insertion: FAILED")
        
        # Unstructured data insertion results
        vector_store_results = results['vector_store_results']
        if results['vector_store_success']:
            logger.info("✅ Unstructured Data Insertion: SUCCESS")
            logger.info(f"📊 Unstructured Data Results:")
            logger.info(f"   Total Documents: {vector_store_results['total']}")
            logger.info(f"   Successful: {vector_store_results['successful']}")
            logger.info(f"   Failed: {vector_store_results['failed']}")
            
            if vector_store_results['total'] > 0:
                success_rate = (vector_store_results['successful'] / vector_store_results['total']) * 100
                logger.info(f"   Success Rate: {success_rate:.1f}%")
        else:
            logger.info("❌ Unstructured Data Insertion: FAILED")
        
        # Overall result
        if results['parsing_success'] and results['insertion_success'] and results['vector_store_success']:
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            logger.info("⚠️  PIPELINE COMPLETED WITH ERRORS")


def main():
    """Main function to run the complete pipeline."""
    
    # Initialize pipeline - inserter will handle its own credentials
    pipeline = WellWisePipeline()
    
    # Check command line arguments
    skip_parsing = '--skip-parsing' in sys.argv
    skip_insertion = '--skip-insertion' in sys.argv
    
    if skip_parsing and skip_insertion:
        logger.error("❌ Cannot skip both parsing and insertion")
        return
    
    # Run the complete pipeline
    results = pipeline.run_complete_pipeline(
        skip_parsing=skip_parsing,
        skip_insertion=skip_insertion
    )
    
    # Exit with appropriate code
    if results['parsing_success'] and results['insertion_success']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main() 