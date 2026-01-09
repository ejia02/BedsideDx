#!/bin/bash
# MongoDB Setup Script for McGee EBM Application

set -e  # Exit on error

echo "🩺 MongoDB Setup for McGee EBM Physical Exam Strategist"
echo "========================================================"
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is not installed. Please install it first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "✅ Homebrew found"
echo ""

# Install MongoDB Community Edition
echo "📦 Installing MongoDB Community Edition..."
brew tap mongodb/brew
brew install mongodb-community

echo ""
echo "✅ MongoDB installed successfully!"
echo ""

# Create data directory if it doesn't exist
DATA_DIR="/usr/local/var/mongodb"
LOG_DIR="/usr/local/var/log/mongodb"

if [ ! -d "$DATA_DIR" ]; then
    echo "📁 Creating MongoDB data directory..."
    mkdir -p "$DATA_DIR"
    echo "✅ Created: $DATA_DIR"
fi

if [ ! -d "$LOG_DIR" ]; then
    echo "📁 Creating MongoDB log directory..."
    mkdir -p "$LOG_DIR"
    echo "✅ Created: $LOG_DIR"
fi

echo ""
echo "🚀 Starting MongoDB service..."
brew services start mongodb-community

echo ""
echo "⏳ Waiting for MongoDB to start (10 seconds)..."
sleep 10

# Test MongoDB connection
echo ""
echo "🧪 Testing MongoDB connection..."
if mongosh --eval "db.adminCommand('ping')" --quiet &> /dev/null; then
    echo "✅ MongoDB is running and accessible!"
    echo ""
    echo "📊 MongoDB Status:"
    mongosh --eval "db.version()" --quiet
    echo ""
    echo "✅ Setup complete! MongoDB is ready to use."
    echo ""
    echo "Next steps:"
    echo "1. Load data into MongoDB:"
    echo "   cd /Users/ericjia/Downloads/BedsideDx"
    echo "   source venv/bin/activate"
    echo "   python mcgee_app/load_sample.py"
    echo ""
    echo "2. Run your application:"
    echo "   streamlit run mcgee_app/app.py --server.port 8503"
else
    echo "⚠️  MongoDB started but connection test failed."
    echo "   Try running manually: mongosh --eval \"db.adminCommand('ping')\""
    echo ""
    echo "   To check MongoDB status: brew services list | grep mongodb"
    echo "   To view MongoDB logs: tail -f $LOG_DIR/mongo.log"
fi

echo ""
echo "📝 MongoDB Configuration:"
echo "   URI: mongodb://localhost:27017"
echo "   Data Directory: $DATA_DIR"
echo "   Log Directory: $LOG_DIR"
echo ""
echo "🔧 Useful commands:"
echo "   Start MongoDB: brew services start mongodb-community"
echo "   Stop MongoDB:  brew services stop mongodb-community"
echo "   Restart MongoDB: brew services restart mongodb-community"
echo "   Check status: brew services list | grep mongodb"
echo "   Connect via shell: mongosh"


