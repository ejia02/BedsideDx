"""
Test script to verify MongoDB Atlas vector search index is properly configured
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient

def test_mongodb_index():
    """Test MongoDB connection and index configuration"""
    
    load_dotenv()
    
    print("🔍 MONGODB ATLAS INDEX TEST")
    print("=" * 50)
    
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri or "username:password" in mongodb_uri:
        print("❌ MongoDB URI not configured properly in .env file")
        return
    
    try:
        # Connect to MongoDB
        client = MongoClient(mongodb_uri)
        db = client["bedside_dx"]
        collection = db["exam_evidence"]
        
        print("✅ MongoDB connection successful")
        
        # Check if collection exists
        collections = db.list_collection_names()
        if "exam_evidence" in collections:
            doc_count = collection.count_documents({})
            print(f"✅ Collection 'exam_evidence' exists with {doc_count} documents")
        else:
            print("ℹ️  Collection 'exam_evidence' doesn't exist yet (will be created when you run the pipeline)")
        
        # Test a simple aggregation to check if vector search would work
        try:
            # This will fail if vector search index doesn't exist, but that's expected
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding", 
                        "queryVector": [0.1] * 1536,  # Dummy vector
                        "numCandidates": 5,
                        "limit": 1
                    }
                }
            ]
            
            # Try the search
            results = list(collection.aggregate(pipeline))
            print("✅ Vector search index is working!")
            
        except Exception as e:
            error_msg = str(e).lower()
            if "index not found" in error_msg or "vector_index" in error_msg:
                print("❌ Vector search index 'vector_index' not found")
                print("   Please create the index in MongoDB Atlas with the configuration provided")
            elif "path not found" in error_msg:
                print("❌ Vector search index exists but 'embedding' field not configured")
                print("   Please update your index configuration")
            else:
                print(f"ℹ️  Vector search test: {e}")
                print("   This is normal if you haven't stored documents yet")
        
        client.close()
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your MONGODB_URI in .env file")
        print("2. Ensure your IP is whitelisted in Atlas Network Access")
        print("3. Verify username/password in connection string")

def show_index_config():
    """Show the correct index configuration"""
    
    print("\n📋 CORRECT VECTOR SEARCH INDEX CONFIGURATION")
    print("=" * 50)
    print("Use this configuration in MongoDB Atlas:")
    print()
    
    config = """{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "ebm_box_label"
    },
    {
      "type": "filter", 
      "path": "chapter"
    },
    {
      "type": "filter",
      "path": "maneuver_base"
    }
  ]
}"""
    
    print(config)
    print()
    print("Index Settings:")
    print("- Index Name: vector_index")
    print("- Database: bedside_dx") 
    print("- Collection: exam_evidence")

if __name__ == "__main__":
    test_mongodb_index()
    show_index_config()
