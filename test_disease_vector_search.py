#!/usr/bin/env python3
"""
Test script for disease-focused vector search.

This script embeds a query using the same SentenceTransformer model
used during ingestion, then searches against ebm_box_label_embedding
to find semantically similar diseases.

Usage:
    python test_disease_vector_search.py "DVT"
    python test_disease_vector_search.py "heart failure"
    python test_disease_vector_search.py "chest pain" --limit 10
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="Test disease-focused vector search"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Disease name or query to search for (e.g., 'DVT', 'heart failure')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="bedside_dx",
        help="MongoDB database name (default: bedside_dx)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="mcgee_evidence",
        help="MongoDB collection name (default: mcgee_evidence)"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="ebm_label_vector_index",
        help="Vector search index name (default: ebm_label_vector_index)"
    )
    
    args = parser.parse_args()
    
    # Check for MongoDB URI
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("❌ MONGODB_URI environment variable not set")
        print("   Set it in your .env file or export it")
        sys.exit(1)
    
    # Import dependencies
    try:
        from pymongo import MongoClient
        from pymongo.errors import OperationFailure
    except ImportError:
        print("❌ pymongo not installed. Run: pip install pymongo")
        sys.exit(1)
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence_transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)
    
    print("="*70)
    print("Disease-Focused Vector Search Test")
    print("="*70)
    print(f"Query:      '{args.query}'")
    print(f"Collection: {args.database}.{args.collection}")
    print(f"Index:      {args.index}")
    print(f"Limit:      {args.limit}")
    print("="*70)
    
    # Load embedding model (same as used during ingestion)
    print("\n📦 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print(f"   Model: all-MiniLM-L6-v2 (dim={model.get_sentence_embedding_dimension()})")
    
    # Generate query embedding
    print(f"\n🔢 Generating embedding for query: '{args.query}'")
    query_embedding = model.encode(args.query).tolist()
    print(f"   Embedding dimension: {len(query_embedding)}")
    
    # Connect to MongoDB
    print(f"\n🔌 Connecting to MongoDB...")
    try:
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
        client.admin.command('ping')
        print("   ✅ Connected successfully")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        sys.exit(1)
    
    db = client[args.database]
    collection = db[args.collection]
    
    # Check collection has documents
    doc_count = collection.count_documents({})
    print(f"   📊 Collection has {doc_count} documents")
    
    if doc_count == 0:
        print("\n❌ Collection is empty. Run the migration script first:")
        print("   python migrate_ebm_label_embeddings.py --source-collection exam_evidence_free --target-collection mcgee_evidence")
        sys.exit(1)
    
    # Check if documents have ebm_box_label_embedding
    docs_with_embedding = collection.count_documents({"ebm_box_label_embedding": {"$exists": True}})
    print(f"   📊 Documents with ebm_box_label_embedding: {docs_with_embedding}")
    
    if docs_with_embedding == 0:
        print("\n❌ No documents have ebm_box_label_embedding field.")
        print("   Run the migration script to add embeddings.")
        sys.exit(1)
    
    # Run vector search
    print(f"\n🔍 Running vector search...")
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": args.index,
                "path": "ebm_box_label_embedding",
                "queryVector": query_embedding,
                "numCandidates": args.limit * 10,
                "limit": args.limit
            }
        },
        {
            "$addFields": {
                "score": {"$meta": "vectorSearchScore"}
            }
        },
        {
            "$project": {
                "embedding": 0,
                "ebm_box_label_embedding": 0
            }
        }
    ]
    
    try:
        results = list(collection.aggregate(pipeline))
        print(f"   ✅ Found {len(results)} results\n")
        
        # If no results, check if index exists by doing a diagnostic
        if len(results) == 0:
            print("⚠️  No results found! This usually means the vector search index doesn't exist.")
            print("\n   Let's verify with a text search fallback...")
            
            # Try a regex search to confirm data exists
            regex_results = list(collection.find(
                {"source.ebm_box_label": {"$regex": args.query, "$options": "i"}},
                {"source.ebm_box_label": 1, "original_finding": 1}
            ).limit(5))
            
            if regex_results:
                print(f"   ✅ Regex search found {len(regex_results)} documents matching '{args.query}':")
                for r in regex_results:
                    label = r.get("source", {}).get("ebm_box_label", "N/A")
                    print(f"      - {label}")
                print("\n   📋 The data exists! You need to create the vector search index.")
            else:
                print(f"   ℹ️  No documents match '{args.query}' even with regex.")
                print("      Try a different query or check your data.")
            
            print("\n" + "="*70)
            print("HOW TO CREATE THE VECTOR SEARCH INDEX")
            print("="*70)
            print("""
1. Go to MongoDB Atlas: https://cloud.mongodb.com
2. Select your cluster → Click "Atlas Search" tab
3. Click "Create Search Index"
4. Choose "Atlas Vector Search" → "JSON Editor"
5. Select database: bedside_dx
6. Select collection: mcgee_evidence
7. Paste this index definition:

{
  "fields": [
    {
      "type": "vector",
      "path": "ebm_box_label_embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}

8. Set index name to: ebm_label_vector_index
9. Click "Create Search Index"
10. Wait 1-2 minutes for status to show "Active"
11. Re-run this test script
""")
            client.close()
            sys.exit(0)
        
    except OperationFailure as e:
        error_msg = str(e).lower()
        if "index not found" in error_msg or "vector" in error_msg:
            print(f"\n❌ Vector search index '{args.index}' not found!")
            print("\n   Create the index in MongoDB Atlas:")
            print("   1. Go to Atlas > Database > Search Indexes")
            print("   2. Create a new Atlas Vector Search index")
            print("   3. Use this configuration:")
            print("""
   {
     "fields": [
       {
         "type": "vector",
         "path": "ebm_box_label_embedding",
         "numDimensions": 384,
         "similarity": "cosine"
       }
     ]
   }
""")
            print(f"   4. Name it: {args.index}")
            print(f"   5. Select collection: {args.collection}")
        else:
            print(f"\n❌ Vector search failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    # Display results
    print("="*70)
    print(f"SEARCH RESULTS for '{args.query}'")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        source = result.get("source", {})
        ebm_box_label = source.get("ebm_box_label") or result.get("ebm_box_label", "N/A")
        original_finding = result.get("original_finding", "N/A")
        score = result.get("score", 0)
        
        # Get LR values
        result_buckets = result.get("result_buckets", [{}])
        bucket = result_buckets[0] if result_buckets else {}
        lr_pos = bucket.get("lr_positive")
        lr_neg = bucket.get("lr_negative")
        
        print(f"\n{i}. {ebm_box_label}")
        print(f"   Score: {score:.4f}")
        print(f"   Finding: {original_finding[:80]}{'...' if len(original_finding) > 80 else ''}")
        if lr_pos is not None:
            print(f"   LR+: {lr_pos}")
        if lr_neg is not None:
            print(f"   LR-: {lr_neg}")
    
    print("\n" + "="*70)
    print("✅ Vector search test complete!")
    print("="*70)
    
    # Show some example queries to try
    print("\nTry these example queries to test semantic matching:")
    print("  python test_disease_vector_search.py 'DVT'")
    print("  python test_disease_vector_search.py 'blood clot in leg'")
    print("  python test_disease_vector_search.py 'CHF'")
    print("  python test_disease_vector_search.py 'heart failure'")
    print("  python test_disease_vector_search.py 'pneumonia'")
    
    client.close()


if __name__ == "__main__":
    main()
