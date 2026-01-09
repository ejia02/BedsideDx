# MongoDB Atlas Setup Guide for Physical Exam Vector Search

## 1. Create MongoDB Atlas Account

1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Sign up for a free account
3. Create a new cluster (M0 free tier is sufficient for testing)

## 2. Set Up Database Access

1. In Atlas dashboard, go to "Database Access"
2. Click "Add New Database User"
3. Create a user with "Read and write to any database" permissions
4. Note down the username and password

## 3. Set Up Network Access

1. Go to "Network Access"
2. Click "Add IP Address"
3. For testing, you can use "Allow Access from Anywhere" (0.0.0.0/0)
4. For production, use your specific IP address

## 4. Get Connection String

1. Go to "Clusters"
2. Click "Connect" on your cluster
3. Choose "Connect your application"
4. Copy the connection string (should look like):
   ```
   mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
   ```

## 5. Create Vector Search Index

**IMPORTANT**: Vector search requires MongoDB Atlas (not self-hosted MongoDB)

### Option A: Atlas UI
1. Go to your cluster in Atlas
2. Click on "Search" tab
3. Click "Create Search Index"
4. Choose "JSON Editor"
5. Use this index definition:

```json
{
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
}
```

6. Set index name to: `vector_index`
7. Set database to: `bedside_dx`
8. Set collection to: `exam_evidence`

### Option B: Atlas CLI
```bash
atlas clusters search indexes create \
  --clusterName <your-cluster-name> \
  --file vector_index.json
```

Where `vector_index.json` contains the index definition above.

## 6. Environment Setup

1. Copy `env_example.txt` to `.env`
2. Fill in your values:
   ```
   OPENAI_API_KEY=sk-your-openai-key
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
   ```

## 7. Test Connection

Run the test script:
```bash
python vector_store.py
```

## Troubleshooting

### Vector Search Not Working
- Ensure you're using MongoDB Atlas (not self-hosted)
- Verify the vector search index is created and active
- Check that the index name matches in your code (`vector_index`)
- Ensure your cluster supports vector search (Atlas M10+ recommended for production)

### Connection Issues
- Check your IP is whitelisted in Network Access
- Verify username/password in connection string
- Ensure connection string format is correct

### Embedding Issues
- Verify OpenAI API key is valid and has credits
- Check rate limits if getting errors
- Ensure you're using the correct model name (`text-embedding-3-small`)

## Performance Notes

- Free tier (M0) has limitations on vector search performance
- For production, consider M10+ clusters
- Vector search index creation can take several minutes
- Batch embedding generation to avoid rate limits

