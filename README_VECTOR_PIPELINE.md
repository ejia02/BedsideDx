# Physical Exam Evidence Vector Pipeline

A complete pipeline for converting physical exam evidence from Excel tables into a searchable vector database using MongoDB Atlas and OpenAI embeddings.

## Features

- **Excel Parsing**: Extracts structured data from physical exam evidence tables
- **Smart Text Processing**: Separates exam maneuvers from result modifiers
- **FREE Vector Embeddings**: Uses local SentenceTransformers (`all-MiniLM-L6-v2`) - no API costs!
- **MongoDB Storage**: Stores documents with embeddings in MongoDB Atlas
- **Dual Vector Search**: 
  - Full-text semantic search on maneuvers and findings
  - Disease-focused semantic search for differential diagnosis matching
- **Batch Processing**: Handles large datasets efficiently

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_vector.txt
```

### 2. Set Up MongoDB Atlas

Follow the detailed guide in `MONGODB_SETUP.md` to:
- Create MongoDB Atlas account
- Set up cluster and database access
- Create vector search index
- Get connection string

### 3. Configure Environment

Create a `.env` file (copy from `env_example.txt`):

```bash
OPENAI_API_KEY=sk-your-openai-key-here
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
```

### 4. Run the Pipeline

```bash
# Run complete pipeline (parse Excel + store in MongoDB)
python physical_exam_pipeline.py full --clear

# Or run individual steps:
python physical_exam_pipeline.py setup --clear  # Setup vector store
python excel_parser.py                          # Test Excel parsing
python vector_store.py                          # Test vector store
```

### 5. Search the Database

```bash
# Basic search
python physical_exam_pipeline.py search --query "heart murmur aortic stenosis"

# Search with filters
python physical_exam_pipeline.py search \
  --query "blood pressure" \
  --ebm-box "Aortic dissection" \
  --limit 5

# Get collection statistics
python physical_exam_pipeline.py stats
```

## Architecture

### Data Flow

1. **Excel Parsing** (`excel_parser.py`)
   - Reads Excel file row by row
   - Identifies chapters and EBM boxes
   - Extracts maneuvers, results, and likelihood ratios
   - Creates structured documents

2. **Embedding Generation** (`vector_store.py`)
   - Generates natural language summaries
   - Creates embeddings using OpenAI API
   - Batches requests for efficiency

3. **MongoDB Storage** (`vector_store.py`)
   - Stores documents with embeddings
   - Creates vector search indexes
   - Supports filtering and metadata queries

4. **Vector Search** (`vector_store.py`)
   - Semantic similarity search
   - Optional filtering by diagnosis/chapter/maneuver
   - Returns ranked results with scores

### Document Structure

Each document represents one exam maneuver result (nested schema):

```python
{
    "source": {
        "ebm_box_label": "Aortic stenosis murmur",
        "chapter": "Chapter 44 Aortic stenosis",
        "ebm_box_id": "44.1"
    },
    "maneuver": {
        "name": "Auscultation of heart",
        "normalized": "auscultation of heart"
    },
    "result_buckets": [
        {
            "label": "systolic murmur grade 3 or louder",
            "lr_positive": 8.2,
            "lr_negative": 0.6,
            "pretest_prob": 15.0
        }
    ],
    "text_for_embedding": "Diagnosis: Aortic stenosis murmur. Maneuver: Auscultation of heart...",
    "original_finding": "Auscultation of heart, systolic murmur grade 3 or louder",
    "embedding": [0.123, -0.456, ...],              # 384-dimensional vector (full text)
    "ebm_box_label_embedding": [0.789, -0.012, ...] # 384-dimensional vector (disease name only)
}
```

### Vector Search Architecture

The system uses two types of vector search:

1. **Full-text Vector Search** (`embedding` field):
   - Searches against the combined text of diagnosis + maneuver + LR values
   - Index name: `vector_index_free`
   - Used for general semantic search across all content

2. **Disease-Focused Vector Search** (`ebm_box_label_embedding` field):
   - Searches only against disease/diagnosis names
   - Index name: `ebm_label_vector_index`
   - Enables semantic matching of abbreviations and synonyms:
     - "DVT" → "Deep Vein Thrombosis"
     - "CHF" → "Heart Failure"
     - "MI" → "Myocardial Infarction"
   - Used when searching by differential diagnosis

## API Reference

### ExcelParser

```python
from excel_parser import ExcelParser

parser = ExcelParser('path/to/excel/file.xlsx')
documents = parser.parse_documents()
```

### VectorStore

```python
from vector_store import VectorStore

# Initialize
store = VectorStore(
    mongodb_uri="mongodb+srv://...",
    openai_api_key="sk-..."
)

# Store documents
store.store_documents(documents)

# Search
results = store.vector_search(
    query="heart murmur",
    limit=10,
    ebm_box_filter="Aortic stenosis"
)
```

### PhysicalExamPipeline

```python
from physical_exam_pipeline import PhysicalExamPipeline

# Initialize and run full pipeline
pipeline = PhysicalExamPipeline('path/to/excel.xlsx')
pipeline.run_full_pipeline(clear_existing=True)

# Search
results = pipeline.search("chest pain", limit=5)
```

## Command Line Usage

```bash
# Full pipeline
python physical_exam_pipeline.py full [--clear] [--batch-size 50]

# Search
python physical_exam_pipeline.py search --query "QUERY" [OPTIONS]
  --limit N              Number of results (default: 10)
  --ebm-box FILTER       Filter by EBM box label
  --chapter FILTER       Filter by chapter
  --maneuver FILTER      Filter by maneuver

# Setup/maintenance
python physical_exam_pipeline.py setup [--clear]
python physical_exam_pipeline.py stats
```

## Performance Considerations

### Embedding Generation
- Uses batch processing (default: 50 documents per batch)
- Respects OpenAI rate limits
- Total time for ~1300 documents: ~10-15 minutes

### MongoDB Atlas
- Free tier (M0) supports vector search but with limitations
- M10+ recommended for production workloads
- Vector index creation takes 2-5 minutes

### Search Performance
- Vector search: ~100-500ms per query
- Filtering adds minimal overhead
- Results ranked by semantic similarity

## Troubleshooting

### Common Issues

1. **"No module named 'sentence_transformers'"**
   ```bash
   pip install -r requirements_vector.txt
   ```

2. **"Vector search failed"**
   - Ensure MongoDB Atlas cluster (not self-hosted)
   - Verify BOTH vector search indexes are created and active:
     - `vector_index_free` (for main embedding)
     - `ebm_label_vector_index` (for disease label embedding)
   - Check index names match in your code

3. **"Disease search not finding results"**
   - Verify `ebm_label_vector_index` exists in Atlas
   - Run the migration script to add `ebm_box_label_embedding` to existing documents:
     ```bash
     python migrate_ebm_label_embeddings.py
     ```

4. **"Connection failed"**
   - Check MongoDB connection string
   - Verify IP whitelist in Atlas
   - Ensure username/password are correct

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Data Sources

The pipeline is designed for McGee's "Evidence-Based Physical Diagnosis" appendix tables, but can be adapted for similar structured medical evidence tables.

Expected Excel format:
- Column 0: Finding/maneuver description
- Column 1: Positive LR with confidence interval
- Column 2: Negative LR with confidence interval  
- Column 3: Pretest probability range
- Chapter headers: "Chapter X ..."
- EBM box headers: "EBM Box X.Y ..."

## License

This pipeline is for educational and research purposes. Ensure you have appropriate rights to use the source data.

