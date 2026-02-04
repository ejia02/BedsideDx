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

## 5. Create Vector Search Indexes

**IMPORTANT**: Vector search requires MongoDB Atlas (not self-hosted MongoDB)

You need to create **TWO** vector search indexes:

### Index 1: Main Vector Search Index (vector_index_free)

This index is used for searching by full text embedding (maneuvers, findings, etc.)

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
      "numDimensions": 384,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "source.ebm_box_label"
    },
    {
      "type": "filter", 
      "path": "source.chapter"
    },
    {
      "type": "filter",
      "path": "maneuver.name"
    }
  ]
}
```

6. Set index name to: `vector_index_free`
7. Set database to: `bedside_dx`
8. Set collection to: `mcgee_evidence`

### Index 2: Disease Label Vector Search Index (ebm_label_vector_index)

This index enables **disease-focused semantic search**. It allows queries like "DVT" to match "Deep Vein Thrombosis" through vector similarity on the disease/diagnosis labels.

1. Click "Create Search Index" again
2. Choose "JSON Editor"
3. Use this index definition:

```json
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
```

4. Set index name to: `ebm_label_vector_index`
5. Set database to: `bedside_dx`
6. Set collection to: `mcgee_evidence`

### Atlas CLI Option
```bash
# Create main vector index
atlas clusters search indexes create \
  --clusterName <your-cluster-name> \
  --file vector_index_free.json

# Create disease label vector index  
atlas clusters search indexes create \
  --clusterName <your-cluster-name> \
  --file ebm_label_vector_index.json
```

### Migration for Existing Data

If you have existing documents without the `ebm_box_label_embedding` field, run the migration script:

```bash
# Preview changes (dry run)
python migrate_ebm_label_embeddings.py --dry-run

# Apply changes
python migrate_ebm_label_embeddings.py
```

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
- Verify BOTH vector search indexes are created and active:
  - `vector_index_free` (for main embedding)
  - `ebm_label_vector_index` (for disease label embedding)
- Check that the index names match in your code
- Ensure your cluster supports vector search (Atlas M10+ recommended for production)

### Disease Search Not Finding Results
- Verify `ebm_label_vector_index` exists in Atlas
- Run the migration script to add `ebm_box_label_embedding` to existing documents:
  ```bash
  python migrate_ebm_label_embeddings.py
  ```
- Check documents have the `ebm_box_label_embedding` field

### Connection Issues
- Check your IP is whitelisted in Network Access
- Verify username/password in connection string
- Ensure connection string format is correct

### Embedding Issues
- This project uses FREE local embeddings via SentenceTransformers (no API key needed!)
- The model `all-MiniLM-L6-v2` generates 384-dimensional embeddings
- Ensure sentence-transformers is installed: `pip install sentence-transformers`

## Performance Notes

- Free tier (M0) has limitations on vector search performance
- For production, consider M10+ clusters
- Vector search index creation can take several minutes
- Embeddings are generated locally using SentenceTransformers (FREE!)

## Vector Search Architecture

The system uses two types of vector search:

1. **Full-text Vector Search** (`embedding` field):
   - Searches against the combined text of diagnosis + maneuver + LR values
   - Used for general semantic search across all content

2. **Disease-Focused Vector Search** (`ebm_box_label_embedding` field):
   - Searches only against disease/diagnosis names
   - Enables semantic matching of abbreviations and synonyms:
     - "DVT" → "Deep Vein Thrombosis"
     - "CHF" → "Heart Failure"
     - "MI" → "Myocardial Infarction"
   - Used when searching by differential diagnosis

