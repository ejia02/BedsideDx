# MongoDB: Optional but Helpful

## TL;DR: You Don't Need MongoDB to Use the App! 🎉

The application is designed to work in **two modes**:

### ✅ **Mode 1: Sample Data (No MongoDB Required)**
- Uses 45+ curated Likelihood Ratio records built into the code
- Works immediately - no setup needed
- Perfect for testing, learning, and demos
- **This is what you're using right now**

### 🗄️ **Mode 2: Full MongoDB Database (Optional)**
- Stores hundreds or thousands of LR records from the full McGee PDF
- Requires MongoDB installation and setup
- Needed only if you want the complete dataset

---

## Why Is MongoDB Not Connecting?

The app is trying to connect to: `mongodb://localhost:27017`

This will fail if:
1. ❌ MongoDB is not installed on your computer
2. ❌ MongoDB is installed but not running
3. ❌ MongoDB is running on a different port
4. ❌ MongoDB is running but requires authentication

**This is completely fine!** The app automatically detects this and switches to sample data mode.

---

## Current Status: What's Happening

When you run the app, here's what happens:

```
1. App tries to connect to MongoDB → ❌ Fails (not installed/running)
2. App automatically detects failure
3. App switches to sample data mode → ✅ Works perfectly!
4. You see: "📝 Using sample data (MongoDB not available)"
5. App functions normally with 45+ LR records
```

**You don't need to do anything!** The app is working as designed.

---

## When Would You Want MongoDB?

You'd only need MongoDB if:

1. **You want the full dataset** (hundreds of LR records instead of 45)
2. **You extracted data from the PDF** using the data ingestion pipeline
3. **You want to add your own LR data** to the database

For most use cases, the sample data is sufficient!

---

## If You Want to Set Up MongoDB (Optional)

### Option A: Local MongoDB (Simpler)

**1. Install MongoDB:**
```bash
# macOS
brew install mongodb-community

# Or download from: https://www.mongodb.com/try/download/community
```

**2. Start MongoDB:**
```bash
# macOS (if installed via Homebrew)
brew services start mongodb-community

# Or run manually:
mongod --dbpath /usr/local/var/mongodb
```

**3. Load Sample Data:**
```bash
cd /Users/ericjia/Downloads/BedsideDx
source venv/bin/activate
python mcgee_app/load_sample.py
```

**4. The app will now connect to MongoDB automatically!**

### Option B: MongoDB Atlas (Cloud - Free Tier)

**1. Create free account:**
- Go to https://www.mongodb.com/atlas
- Sign up for free M0 tier

**2. Get connection string:**
- Create a cluster
- Get connection string like: `mongodb+srv://user:pass@cluster.mongodb.net`

**3. Update config:**
```bash
export MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net"
```

**4. Load data:**
```bash
python mcgee_app/load_sample.py
```

---

## Summary

| Feature | Sample Data Mode | MongoDB Mode |
|---------|------------------|--------------|
| **Setup Required** | ❌ None | ✅ Install/configure MongoDB |
| **Number of Records** | 45+ | Hundreds/thousands |
| **Works Immediately** | ✅ Yes | ❌ Needs setup |
| **Good For** | Testing, demos, learning | Production, full dataset |
| **Your Current Status** | ✅ **This is what you're using!** | ❌ Not needed |

---

## Bottom Line

**You're all set!** The app is working correctly with sample data mode. MongoDB is only needed if you want to expand beyond the 45 built-in LR records. The app handles everything automatically - no action needed on your part! 🎉


