#!/bin/bash
# Start Streamlit Application Script

echo "🩺 Starting McGee EBM Physical Exam Strategist"
echo "================================================"
echo ""

# Navigate to project directory
cd /Users/ericjia/Downloads/BedsideDx

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Creating virtual environment..."
    python3 -m venv venv
    echo "   Installing dependencies..."
    source venv/bin/activate
    pip install -r mcgee_app/requirements.txt
else
    echo "✅ Virtual environment found"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Check if .env file exists
if [ -f ".env" ]; then
    echo "✅ Found .env file"
else
    echo "⚠️  No .env file found"
    echo "   Make sure to create .env with OPENAI_API_KEY and MONGODB_URI"
fi

# Test MongoDB connection
echo ""
echo "🧪 Testing MongoDB connection..."
python -c "
import sys
sys.path.insert(0, 'mcgee_app')
try:
    from config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME
    from pymongo import MongoClient
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    count = collection.count_documents({})
    print(f'✅ MongoDB connected successfully!')
    print(f'   URI: {MONGODB_URI[:50]}...')
    print(f'   Database: {DATABASE_NAME}')
    print(f'   Collection: {COLLECTION_NAME}')
    print(f'   Records: {count}')
    client.close()
except Exception as e:
    print(f'❌ MongoDB connection failed: {str(e)[:100]}')
    print('   The app will show an error. Please check your MongoDB setup.')
    sys.exit(1)
" || echo "⚠️  MongoDB connection test failed"

# Find available port
PORT=8503
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    PORT=$((PORT + 1))
done

echo ""
echo "🚀 Starting Streamlit on port $PORT..."
echo "   URL: http://localhost:$PORT"
echo ""
echo "📝 Press Ctrl+C to stop the server"
echo ""

# Start Streamlit
streamlit run mcgee_app/app.py --server.port $PORT

