"""
Migration Script: Add ebm_box_label_embedding to Existing Documents

This script migrates documents from a source collection to a target collection,
adding the ebm_box_label_embedding field for disease-focused semantic search.

The ebm_box_label_embedding enables semantic matching like:
- "DVT" -> "Deep Vein Thrombosis"
- "CHF" -> "Heart Failure"
- "MI" -> "Myocardial Infarction"

Usage:
    # Migrate from exam_evidence_free to mcgee_evidence (with new embeddings)
    python migrate_ebm_label_embeddings.py \\
        --source-collection exam_evidence_free \\
        --target-collection mcgee_evidence \\
        --dry-run

    # Update documents in place (same source and target)
    python migrate_ebm_label_embeddings.py \\
        --source-collection mcgee_evidence \\
        --target-collection mcgee_evidence

Options:
    --dry-run             Preview changes without actually updating documents
    --batch-size          Number of documents to process per batch (default: 100)
    --source-collection   Source collection to read from
    --target-collection   Target collection to write to (can be same as source)
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
except ImportError:
    logger.error("pymongo not installed. Run: pip install pymongo")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("sentence_transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)


def connect_to_mongodb(uri: str, database_name: str):
    """Connect to MongoDB and return client and database reference."""
    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=20000,
            connectTimeoutMS=20000
        )
        # Test connection
        client.admin.command('ping')
        logger.info(f"Connected to MongoDB database: {database_name}")
        
        db = client[database_name]
        return client, db
        
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load the SentenceTransformer embedding model."""
    logger.info(f"Loading embedding model: {model_name}")
    try:
        # Force CPU to avoid MPS compatibility issues on Apple Silicon
        model = SentenceTransformer(model_name, device="cpu")
        logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        sys.exit(1)


def get_all_documents(collection, batch_size: int = 100):
    """
    Get all documents from collection.
    
    Returns cursor for memory efficiency.
    """
    # Exclude existing embeddings from fetch to save bandwidth
    projection = {
        "embedding": 0,
        "ebm_box_label_embedding": 0
    }
    
    cursor = collection.find({}, projection).batch_size(batch_size)
    return cursor


def migrate_documents(
    source_collection,
    target_collection,
    embedding_model: SentenceTransformer,
    batch_size: int = 100,
    dry_run: bool = False,
    same_collection: bool = False
) -> Dict[str, int]:
    """
    Migrate documents from source to target collection, adding ebm_box_label_embedding.
    
    Args:
        source_collection: MongoDB collection to read from
        target_collection: MongoDB collection to write to
        embedding_model: SentenceTransformer model for generating embeddings
        batch_size: Number of documents to process per batch
        dry_run: If True, only preview changes without updating
        same_collection: If True, update in place instead of inserting
        
    Returns:
        Statistics dictionary with counts
    """
    stats = {
        "total_processed": 0,
        "migrated": 0,
        "skipped": 0,
        "errors": 0
    }
    
    # Count total documents in source
    total_docs = source_collection.count_documents({})
    logger.info(f"Found {total_docs} documents in source collection")
    
    if total_docs == 0:
        logger.info("No documents found in source collection.")
        return stats
    
    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    
    # Process in batches
    cursor = get_all_documents(source_collection, batch_size)
    batch = []
    batch_num = 0
    
    for doc in cursor:
        batch.append(doc)
        
        if len(batch) >= batch_size:
            batch_num += 1
            batch_stats = process_batch(
                source_collection, target_collection, embedding_model, 
                batch, batch_num, dry_run, same_collection
            )
            
            stats["total_processed"] += batch_stats["processed"]
            stats["migrated"] += batch_stats["migrated"]
            stats["skipped"] += batch_stats["skipped"]
            stats["errors"] += batch_stats["errors"]
            
            # Progress update
            progress = (stats["total_processed"] / total_docs) * 100
            logger.info(f"Progress: {stats['total_processed']}/{total_docs} ({progress:.1f}%)")
            
            batch = []
    
    # Process remaining documents
    if batch:
        batch_num += 1
        batch_stats = process_batch(
            source_collection, target_collection, embedding_model,
            batch, batch_num, dry_run, same_collection
        )
        
        stats["total_processed"] += batch_stats["processed"]
        stats["migrated"] += batch_stats["migrated"]
        stats["skipped"] += batch_stats["skipped"]
        stats["errors"] += batch_stats["errors"]
    
    return stats


def process_batch(
    source_collection,
    target_collection,
    embedding_model: SentenceTransformer,
    batch: List[Dict[str, Any]],
    batch_num: int,
    dry_run: bool,
    same_collection: bool
) -> Dict[str, int]:
    """Process a batch of documents."""
    stats = {"processed": 0, "migrated": 0, "skipped": 0, "errors": 0}
    
    # Extract ebm_box_labels for batch embedding
    labels = []
    valid_docs = []
    
    for doc in batch:
        stats["processed"] += 1
        
        # Get ebm_box_label from nested schema (try both formats)
        source = doc.get("source", {})
        ebm_box_label = source.get("ebm_box_label") or doc.get("ebm_box_label")
        
        if not ebm_box_label:
            logger.warning(f"Document {doc['_id']} has no ebm_box_label, skipping")
            stats["skipped"] += 1
            continue
        
        labels.append(ebm_box_label)
        valid_docs.append(doc)
    
    if not valid_docs:
        return stats
    
    # Generate embeddings for all labels in batch
    try:
        logger.info(f"Batch {batch_num}: Generating embeddings for {len(labels)} labels...")
        embeddings = embedding_model.encode(labels)
        
    except Exception as e:
        logger.error(f"Batch {batch_num}: Failed to generate embeddings: {e}")
        stats["errors"] += len(valid_docs)
        return stats
    
    # Process documents
    for doc, embedding in zip(valid_docs, embeddings):
        try:
            if dry_run:
                source = doc.get("source", {})
                ebm_label = source.get("ebm_box_label") or doc.get("ebm_box_label", "N/A")
                logger.info(f"  Would migrate: {ebm_label[:50]}...")
                stats["migrated"] += 1
            else:
                if same_collection:
                    # Update in place
                    result = target_collection.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {"ebm_box_label_embedding": embedding.tolist()}}
                    )
                    if result.modified_count > 0:
                        stats["migrated"] += 1
                    else:
                        stats["skipped"] += 1
                else:
                    # Insert into target collection (remove _id to get new one)
                    new_doc = doc.copy()
                    new_doc.pop("_id", None)  # Remove old _id
                    new_doc["ebm_box_label_embedding"] = embedding.tolist()
                    
                    # Re-fetch the full document with embedding field if needed
                    full_doc = source_collection.find_one({"_id": doc["_id"]})
                    if full_doc:
                        full_doc.pop("_id", None)
                        full_doc["ebm_box_label_embedding"] = embedding.tolist()
                        target_collection.insert_one(full_doc)
                        stats["migrated"] += 1
                    else:
                        stats["errors"] += 1
                    
        except Exception as e:
            logger.error(f"Failed to migrate document {doc['_id']}: {e}")
            stats["errors"] += 1
    
    logger.info(f"Batch {batch_num}: Migrated {stats['migrated']}, Skipped {stats['skipped']}, Errors {stats['errors']}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate MongoDB documents to add ebm_box_label_embedding field"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually updating documents"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents to process per batch (default: 100)"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="bedside_dx",
        help="MongoDB database name (default: bedside_dx)"
    )
    parser.add_argument(
        "--source-collection",
        type=str,
        default="exam_evidence_free",
        help="Source collection to read from (default: exam_evidence_free)"
    )
    parser.add_argument(
        "--target-collection",
        type=str,
        default="mcgee_evidence",
        help="Target collection to write to (default: mcgee_evidence)"
    )
    
    args = parser.parse_args()
    
    # Get MongoDB URI from environment
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        logger.error("MONGODB_URI environment variable not set")
        logger.error("Set it in your .env file or export it: export MONGODB_URI='your-uri'")
        sys.exit(1)
    
    same_collection = args.source_collection == args.target_collection
    
    print("="*60)
    print("EBM Box Label Embedding Migration")
    print("="*60)
    print(f"Database:          {args.database}")
    print(f"Source Collection: {args.source_collection}")
    print(f"Target Collection: {args.target_collection}")
    print(f"Mode:              {'Update in place' if same_collection else 'Copy to new collection'}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Dry Run:           {args.dry_run}")
    print("="*60)
    
    # Connect to MongoDB
    client, db = connect_to_mongodb(mongodb_uri, args.database)
    
    source_collection = db[args.source_collection]
    target_collection = db[args.target_collection]
    
    # Check source collection has documents
    source_count = source_collection.count_documents({})
    print(f"\nSource collection '{args.source_collection}' has {source_count} documents")
    
    if not same_collection:
        target_count = target_collection.count_documents({})
        print(f"Target collection '{args.target_collection}' has {target_count} documents")
        
        if target_count > 0 and not args.dry_run:
            response = input(f"\nTarget collection already has {target_count} documents. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
    
    # Load embedding model
    embedding_model = load_embedding_model()
    
    # Run migration
    stats = migrate_documents(
        source_collection,
        target_collection,
        embedding_model,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        same_collection=same_collection
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Migration Summary")
    print("="*60)
    print(f"Total Processed: {stats['total_processed']}")
    print(f"Migrated:        {stats['migrated']}")
    print(f"Skipped:         {stats['skipped']}")
    print(f"Errors:          {stats['errors']}")
    
    if args.dry_run:
        print("\nThis was a DRY RUN - no changes were made.")
        print("Run without --dry-run to apply changes.")
    else:
        print("\nMigration complete!")
        print(f"\nDocuments are now in: {args.database}.{args.target_collection}")
        print("\nIMPORTANT: Don't forget to create the vector search index in MongoDB Atlas:")
        print(f"""
In MongoDB Atlas > Database > {args.database} > {args.target_collection} > Search Indexes:

{{
  "name": "ebm_label_vector_index",
  "fields": [
    {{
      "type": "vector",
      "path": "ebm_box_label_embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }}
  ]
}}
""")
    
    # Close connection
    client.close()


if __name__ == "__main__":
    main()
