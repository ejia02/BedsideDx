"""
FREE Physical Exam Evidence Pipeline
Uses local embeddings instead of OpenAI (no API costs!)
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional

from excel_parser import ExcelParser, ExamDocument
from vector_store_free import FreeVectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FreePhysicalExamPipeline:
    """FREE pipeline for physical exam evidence processing"""
    
    def __init__(self, 
                 excel_path: str,
                 mongodb_uri: str = None,
                 database_name: str = "bedside_dx",
                 collection_name: str = "mcgee_evidence"):
        
        self.excel_path = excel_path
        self.parser = ExcelParser(excel_path)
        self.vector_store = FreeVectorStore(
            mongodb_uri=mongodb_uri,
            database_name=database_name,
            collection_name=collection_name
        )
        self.documents: List[ExamDocument] = []
    
    def load_and_parse_excel(self) -> List[ExamDocument]:
        """Load and parse Excel file"""
        logger.info(f"Loading Excel file: {self.excel_path}")
        self.documents = self.parser.parse_documents()
        logger.info(f"Parsed {len(self.documents)} documents")
        return self.documents
    
    def setup_vector_store(self, clear_existing: bool = False) -> None:
        """Set up vector store and optionally clear existing data"""
        if clear_existing:
            logger.info("Clearing existing collection...")
            self.vector_store.clear_collection()
        
        logger.info("Vector search index configuration for FREE model:")
        self.vector_store.create_vector_search_index_info()
    
    def store_documents(self, batch_size: int = 100) -> None:
        """Store documents in vector store with batching (FREE!)"""
        if not self.documents:
            logger.error("No documents loaded. Run load_and_parse_excel() first.")
            return
        
        logger.info(f"Storing {len(self.documents)} documents using FREE local embeddings")
        logger.info("This may take a few minutes but costs nothing!")
        
        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(self.documents) - 1) // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            self.vector_store.store_documents(batch)
    
    def search(self, 
               query: str, 
               limit: int = 10,
               ebm_box_filter: Optional[str] = None,
               chapter_filter: Optional[str] = None,
               maneuver_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search the vector store (FREE!)"""
        return self.vector_store.vector_search(
            query=query,
            limit=limit,
            ebm_box_filter=ebm_box_filter,
            chapter_filter=chapter_filter,
            maneuver_filter=maneuver_filter
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return self.vector_store.get_collection_stats()
    
    def run_full_pipeline(self, clear_existing: bool = False, batch_size: int = 100) -> None:
        """Run the complete FREE pipeline"""
        logger.info("Starting FREE pipeline (no API costs!)...")
        
        # Step 1: Parse Excel
        self.load_and_parse_excel()
        
        # Step 2: Setup vector store
        self.setup_vector_store(clear_existing=clear_existing)
        
        # Step 3: Store documents
        self.store_documents(batch_size=batch_size)
        
        # Step 4: Display stats
        stats = self.get_stats()
        logger.info(f"FREE pipeline complete! Collection stats: {stats}")


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results for display"""
    if not results:
        return "No results found."
    
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"\n--- Result {i} (Score: {result.get('score', 0):.3f}) ---")
        output.append(f"Finding: {result.get('original_finding', 'N/A')}")
        output.append(f"EBM Box: {result.get('ebm_box_label', 'N/A')}")
        output.append(f"Chapter: {result.get('chapter', 'N/A')}")
        output.append(f"Maneuver: {result.get('maneuver_base', 'N/A')}")
        
        if result.get('result_modifier'):
            output.append(f"Result: {result.get('result_modifier')}")
        
        # Likelihood ratios
        pos_lr = result.get('pos_lr_numeric')
        neg_lr = result.get('neg_lr_numeric')
        pretest = result.get('pretest_prob_numeric')
        
        lr_info = []
        if pos_lr is not None:
            lr_info.append(f"LR+: {pos_lr}")
        if neg_lr is not None:
            lr_info.append(f"LR-: {neg_lr}")
        if pretest is not None:
            lr_info.append(f"Pretest: {pretest}%")
        
        if lr_info:
            output.append(f"Metrics: {', '.join(lr_info)}")
    
    return '\n'.join(output)


def main():
    """Command line interface for FREE pipeline"""
    parser = argparse.ArgumentParser(description='FREE Physical Exam Evidence Pipeline')
    parser.add_argument('command', choices=['setup', 'search', 'stats', 'full'], 
                       help='Command to run')
    parser.add_argument('--excel', default='/Users/ericjia/Downloads/BedsideDx/AppendixChp71_Table71_1 (1).xlsx',
                       help='Path to Excel file')
    parser.add_argument('--query', help='Search query (for search command)')
    parser.add_argument('--limit', type=int, default=10, help='Number of search results')
    parser.add_argument('--ebm-box', help='Filter by EBM box label')
    parser.add_argument('--chapter', help='Filter by chapter')
    parser.add_argument('--maneuver', help='Filter by maneuver')
    parser.add_argument('--clear', action='store_true', help='Clear existing data')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Check for MongoDB URI (no OpenAI key needed!)
    if not os.getenv('MONGODB_URI'):
        print("Error: MONGODB_URI environment variable not set")
        print("Please set up your MongoDB connection in .env file")
        print("(No OpenAI API key needed for the FREE version!)")
        sys.exit(1)
    
    # Initialize FREE pipeline
    try:
        pipeline = FreePhysicalExamPipeline(args.excel)
        print("🆓 Using FREE local embeddings (no API costs!)")
    except Exception as e:
        logger.error(f"Failed to initialize FREE pipeline: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'setup':
            logger.info("Setting up FREE vector store...")
            pipeline.setup_vector_store(clear_existing=args.clear)
            
        elif args.command == 'search':
            if not args.query:
                print("Error: --query required for search command")
                sys.exit(1)
            
            logger.info(f"Searching for: '{args.query}' (FREE!)")
            results = pipeline.search(
                query=args.query,
                limit=args.limit,
                ebm_box_filter=args.ebm_box,
                chapter_filter=args.chapter,
                maneuver_filter=args.maneuver
            )
            
            print(format_search_results(results))
            
        elif args.command == 'stats':
            stats = pipeline.get_stats()
            import json
            print(json.dumps(stats, indent=2))
            
        elif args.command == 'full':
            pipeline.run_full_pipeline(
                clear_existing=args.clear,
                batch_size=args.batch_size
            )
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

