"""
Test script to verify API keys are working
"""

import os
from dotenv import load_dotenv

def test_api_keys():
    """Test if API keys are properly configured"""
    
    # Load environment variables
    load_dotenv()
    
    print("🔑 API KEY CONFIGURATION TEST")
    print("=" * 50)
    
    # Test OpenAI API Key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your_openai_api_key_here":
        print(f"✅ OpenAI API Key: Found (starts with: {openai_key[:10]}...)")
        
        # Test OpenAI connection
        try:
            import openai
            openai.api_key = openai_key
            
            # Try a simple API call
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input="test"
            )
            print("✅ OpenAI API: Connection successful!")
            
        except Exception as e:
            print(f"❌ OpenAI API Error: {e}")
            
    else:
        print("❌ OpenAI API Key: Not configured or still using placeholder")
    
    print()
    
    # Test MongoDB URI
    mongodb_uri = os.getenv("MONGODB_URI")
    if mongodb_uri and "username:password" not in mongodb_uri:
        print(f"✅ MongoDB URI: Found (starts with: mongodb+srv://...)")
        
        # Test MongoDB connection
        try:
            from pymongo import MongoClient
            client = MongoClient(mongodb_uri)
            client.admin.command('ping')
            print("✅ MongoDB: Connection successful!")
            client.close()
            
        except Exception as e:
            print(f"❌ MongoDB Error: {e}")
            
    else:
        print("❌ MongoDB URI: Not configured or still using placeholder")
    
    print()
    print("📋 NEXT STEPS:")
    if openai_key == "your_openai_api_key_here":
        print("1. Get OpenAI API key from https://platform.openai.com/api-keys")
    if "username:password" in mongodb_uri:
        print("2. Set up MongoDB Atlas and get connection string")
    
    print("3. Edit .env file with your actual keys")
    print("4. Run: python physical_exam_pipeline.py full --clear")

if __name__ == "__main__":
    test_api_keys()

