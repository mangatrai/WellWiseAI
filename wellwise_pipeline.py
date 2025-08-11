#!/usr/bin/env python3
"""
WellWiseAI Complete Pipeline: Parse and Insert Data
"""

import os
import sys
import logging
from datetime import datetime
from wellwise_parser import main as parse_data
from db.insert_data import WellWiseDBInserter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WellWisePipeline:
    """Complete pipeline for parsing and inserting oil and gas data."""
    
    def __init__(self):
        """
        Initialize the pipeline.
        """
        # Initialize inserter - it will handle its own credentials
        self.inserter = WellWiseDBInserter()
        self.db_available = self.inserter.is_available()
    
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
            'parsing_records': 0,
            'insertion_results': {'successful': 0, 'failed': 0, 'total': 0},
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
        
        # Step 2: Database Insertion
        if not skip_insertion:
            logger.info("=" * 50)
            logger.info("STEP 2: DATABASE INSERTION")
            logger.info("=" * 50)
            
            insertion_results = self.run_database_insertion()
            results['insertion_results'] = insertion_results
            results['insertion_success'] = insertion_results['successful'] > 0
            
            if insertion_results['successful'] == 0:
                logger.error("❌ Pipeline failed at insertion step")
        else:
            logger.info("⏭️  Skipping insertion step")
            results['insertion_success'] = True
        
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
        
        # Insertion results
        insertion_results = results['insertion_results']
        if results['insertion_success']:
            logger.info("✅ Database Insertion: SUCCESS")
            logger.info(f"📊 Insertion Results:")
            logger.info(f"   Total Records: {insertion_results['total']}")
            logger.info(f"   Successful: {insertion_results['successful']}")
            logger.info(f"   Failed: {insertion_results['failed']}")
            
            if insertion_results['total'] > 0:
                success_rate = (insertion_results['successful'] / insertion_results['total']) * 100
                logger.info(f"   Success Rate: {success_rate:.1f}%")
        else:
            logger.info("❌ Database Insertion: FAILED")
        
        # Overall result
        if results['parsing_success'] and results['insertion_success']:
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