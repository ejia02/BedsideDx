# MongoDB Setup Guide - Step by Step

## Quick Start (macOS with Homebrew)

### Step 1: Install MongoDB Community Edition

```bash
# Add MongoDB tap
brew tap mongodb/brew

# Install MongoDB
brew install mongodb-community
```

### Step 2: Start MongoDB Service

```bash
# Start MongoDB (starts automatically on boot)
brew services start mongodb-community

# Verify it's running
brew services list | grep mongodb
```

### Step 3: Verify Installation

```bash
# Test MongoDB connection
mongosh --eval "db.adminCommand('ping')"

# You should see: { ok: 1 }
```

### Step 4: Load Data into MongoDB

```bash
# Navigate to project directory
cd /Users/ericjia/Downloads/BedsideDx

# Activate virtual environment
source venv/bin/activate

# Load sample LR data into MongoDB
python mcgee_app/load_sample.py
```

### Step 5: Test MongoDB Connection

```bash
# Run connection test script
python mcgee_app/test_mongodb_connection.py
```

### Step 6: Run Your Application

```bash
# Start the Streamlit app
streamlit run mcgee_app/app.py --server.port 8503
```

---

## Alternative: Use Setup Script

I've created an automated setup script:

```bash
cd /Users/ericjia/Downloads/BedsideDx
bash mcgee_app/setup_mongodb.sh
```

This script will:
- ✅ Check if Homebrew is installed
- ✅ Install MongoDB Community Edition
- ✅ Create necessary directories
- ✅ Start MongoDB service
- ✅ Test the connection
- ✅ Provide next steps

---

## Verification Checklist

After setup, verify everything works:

- [ ] MongoDB is installed: `brew list mongodb-community`
- [ ] MongoDB is running: `brew services list | grep mongodb`
- [ ] Connection works: `python mcgee_app/test_mongodb_connection.py`
- [ ] Data is loaded: Check script output for record count
- [ ] App connects: Run Streamlit app and check sidebar status

---

## Troubleshooting

### MongoDB Won't Start

```bash
# Check MongoDB status
brew services list | grep mongodb

# View MongoDB logs
tail -f /usr/local/var/log/mongodb/mongo.log

# Try manual start
mongod --config /usr/local/etc/mongod.conf
```

### Connection Refused

```bash
# Verify MongoDB is listening on port 27017
lsof -i :27017

# Check if another MongoDB instance is running
ps aux | grep mongod
```

### Permission Issues

```bash
# Ensure data directory has correct permissions
sudo chown -R $(whoami) /usr/local/var/mongodb
sudo chown -R $(whoami) /usr/local/var/log/mongodb
```

### Data Not Loading

```bash
# Check MongoDB connection
python mcgee_app/test_mongodb_connection.py

# Manually verify data
mongosh
use mcgee_ebm
db.likelihood_ratios.countDocuments()
db.likelihood_ratios.find().limit(5)
```

---

## MongoDB Configuration

**Connection URI:** `mongodb://localhost:27017`

**Database:** `mcgee_ebm`

**Collection:** `likelihood_ratios`

**Data Directory:** `/usr/local/var/mongodb`

**Log Directory:** `/usr/local/var/log/mongodb`

---

## Useful Commands

```bash
# Start MongoDB
brew services start mongodb-community

# Stop MongoDB
brew services stop mongodb-community

# Restart MongoDB
brew services restart mongodb-community

# Connect to MongoDB shell
mongosh

# View MongoDB status
brew services list | grep mongodb

# View logs
tail -f /usr/local/var/log/mongodb/mongo.log
```

---

## Next Steps

Once MongoDB is set up and data is loaded:

1. ✅ Run connection test: `python mcgee_app/test_mongodb_connection.py`
2. ✅ Start application: `streamlit run mcgee_app/app.py --server.port 8503`
3. ✅ Verify in app sidebar: Should show "✅ MongoDB connected" with record count

---

## Alternative: MongoDB Atlas (Cloud)

If you prefer cloud MongoDB instead of local:

1. Sign up at https://www.mongodb.com/atlas (free tier available)
2. Create a cluster
3. Get connection string
4. Update `MONGODB_URI` in `config.py` or set environment variable:
   ```bash
   export MONGODB_URI="mongodb+srv://username:password@cluster.mongodb.net"
   ```
5. Load data: `python mcgee_app/load_sample.py`

---

## Support

If you encounter issues:

1. Check MongoDB logs: `/usr/local/var/log/mongodb/mongo.log`
2. Run test script: `python mcgee_app/test_mongodb_connection.py`
3. Verify MongoDB is running: `brew services list | grep mongodb`
4. Check connection: `mongosh --eval "db.adminCommand('ping')"`


