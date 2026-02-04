#!/usr/bin/env python3
"""
Test MongoDB connection and verify database setup.

This script checks:
1. MongoDB connection status
2. Database and collection existence
3. Data availability
4. Index configuration
"""

import sys
from config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
    import certifi
    PYMONGO_AVAILABLE = True
except ImportError:
    print("❌ pymongo not installed. Install with: pip install pymongo certifi")
    sys.exit(1)

def test_mongodb_connection():
    """Test MongoDB connection and provide detailed status."""
    
    print("🩺 MongoDB Connection Test")
    print("=" * 50)
    print(f"URI: {MONGODB_URI}")
    print(f"Database: {DATABASE_NAME}")
    print(f"Collection: {COLLECTION_NAME}")
    print()
    
    # Test 1: Connection
    print("1️⃣ Testing MongoDB connection...")
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
        # Force connection attempt
        client.admin.command('ping')
        print("   ✅ Connected successfully!")
    except ConnectionFailure as e:
        print(f"   ❌ Connection failed: {e}")
        print()
        print("💡 Troubleshooting:")
        if "SSL" in str(e) or "TLS" in str(e) or "INTERNAL_ERROR" in str(e):
            print("   🔐 SSL/TLS Error detected! Common causes:")
            print("   • Your IP address is NOT whitelisted in MongoDB Atlas")
            print("   • Go to: MongoDB Atlas → Network Access → Add IP Address")
            print("   • Add your current IP or use 0.0.0.0/0 for development")
            print("   • MongoDB Atlas cluster may be paused - check Atlas console")
        else:
            print("   • For local MongoDB: brew install mongodb-community")
            print("   • Start MongoDB: brew services start mongodb-community")
            print("   • Check MongoDB status: brew services list | grep mongodb")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False
    
    # Test 2: Database and Collection
    print()
    print("2️⃣ Checking database and collection...")
    try:
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Check if collection exists
        collection_names = db.list_collection_names()
        if COLLECTION_NAME in collection_names:
            print(f"   ✅ Collection '{COLLECTION_NAME}' exists")
        else:
            print(f"   ⚠️  Collection '{COLLECTION_NAME}' does not exist")
            print(f"   💡 It will be created when data is loaded")
    
    except Exception as e:
        print(f"   ❌ Error checking database: {e}")
        client.close()
        return False
    
    # Test 3: Data Count
    print()
    print("3️⃣ Checking data availability...")
    try:
        count = collection.count_documents({})
        if count > 0:
            print(f"   ✅ Found {count} documents in collection")
        else:
            print(f"   ⚠️  Collection is empty (0 documents)")
            print(f"   💡 Load data with: python mcgee_app/load_sample.py")
            client.close()
            return False
    except Exception as e:
        print(f"   ⚠️  Could not count documents: {e}")
    
    # Test 4: Sample Data
    print()
    print("4️⃣ Checking sample records...")
    try:
        sample = collection.find_one({})
        if sample:
            print("   ✅ Sample record:")
            print(f"      Disease: {sample.get('disease', 'N/A')}")
            print(f"      Finding: {sample.get('finding', 'N/A')[:50]}...")
        else:
            print("   ⚠️  No sample records found")
    except Exception as e:
        print(f"   ⚠️  Could not retrieve sample: {e}")
    
    # Test 5: Indexes
    print()
    print("5️⃣ Checking indexes...")
    try:
        indexes = collection.list_indexes()
        index_list = list(indexes)
        if index_list:
            print(f"   ✅ Found {len(index_list)} index(es):")
            for idx in index_list:
                print(f"      • {idx.get('name', 'unnamed')}")
        else:
            print("   ⚠️  No indexes found (will be created on first data load)")
    except Exception as e:
        print(f"   ⚠️  Could not check indexes: {e}")
    
    # Test 6: Query Test
    print()
    print("6️⃣ Testing query functionality...")
    try:
        test_query = collection.find({"disease": {"$regex": "DVT", "$options": "i"}}).limit(1)
        test_results = list(test_query)
        if test_results:
            print(f"   ✅ Query test successful (found {len(test_results)} result(s))")
        else:
            print("   ⚠️  Query returned no results (database may need more data)")
    except Exception as e:
        print(f"   ⚠️  Query test failed: {e}")
    
    client.close()
    
    print()
    print("=" * 50)
    print("✅ MongoDB connection test complete!")
    print()
    
    return True

if __name__ == "__main__":
    success = test_mongodb_connection()
    sys.exit(0 if success else 1)


