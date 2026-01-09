# McGee EBM Physical Exam Strategist

A production-ready, IRB-compliant educational tool for generating evidence-based physical examination strategies using Likelihood Ratio (LR) data from McGee's Evidence-Based Physical Diagnosis textbook.

## 🎯 Purpose

This application helps medical students and healthcare educators:

1. **Learn evidence-based physical examination** by understanding which maneuvers have the highest diagnostic utility
2. **Identify high-yield maneuvers** (LR+ ≥ 5.0 or LR- ≤ 0.2) that significantly change disease probability
3. **Recognize low-utility maneuvers** (LR 0.5-2.0) that provide minimal diagnostic information
4. **Understand Likelihood Ratios** and their clinical application

## ⚠️ Important Disclaimer

**This tool is for educational purposes only and must not replace clinical judgment or official diagnosis.**

The application is designed to be IRB-compliant by:
- Never providing definitive diagnoses
- Always framing output as educational guidance
- Citing evidence (LRs) for all recommendations
- Including clear disclaimers about clinical limitations

## 🏗️ Architecture

The application uses a **Structured RAG (Retrieval-Augmented Generation)** pipeline:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Differential   │───▶│    MongoDB      │
│   (Symptoms)    │    │   Generation    │    │   LR Retrieval  │
└─────────────────┘    │   (GPT-4o)      │    │                 │
                       └─────────────────┘    └────────┬────────┘
                                                       │
                       ┌─────────────────┐    ┌────────▼────────┐
                       │   Educational   │◀───│   Evidence      │
                       │    Strategy     │    │  Categorization │
                       │   (GPT-4o)      │    │                 │
                       └─────────────────┘    └─────────────────┘
```

### Components

1. **`config.py`** - Configuration management (API keys, database settings, thresholds)
2. **`data_ingestion.py`** - PDF extraction and MongoDB loading pipeline
3. **`rag_engine.py`** - Core RAG pipeline with three main functions:
   - `generate_differential_diagnosis()` - Symptom analysis
   - `retrieve_evidence()` - MongoDB LR data retrieval
   - `synthesize_educational_strategy()` - IRB-compliant output generation
4. **`app.py`** - Streamlit web interface

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- MongoDB (local or cloud instance)
- OpenAI API key
- Poppler (for PDF processing)

### 1. Install Dependencies

```bash
# Navigate to the mcgee_app directory
cd mcgee_app

# Install Python dependencies
pip install -r requirements.txt

# Install Poppler (required for pdf2image)
# macOS:
brew install poppler

# Ubuntu/Debian:
sudo apt-get install poppler-utils

# Windows:
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
```

### 2. Configure Environment

Copy the example environment file and fill in your credentials:

```bash
# From project root
cp .env.example .env
```

Then edit `.env` with your values:

```env
# Required
OPENAI_API_KEY=your-openai-api-key
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net

# Optional (these are the defaults)
DATABASE_NAME=bedside_dx
COLLECTION_NAME=exam_evidence_free
LOG_LEVEL=INFO
```

Alternatively, set environment variables directly:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net"
```

### 3. Initialize Database

#### Option A: Load Sample Data (Quick Start)

```bash
python -m mcgee_app.data_ingestion --sample
```

This loads a representative sample of ~45 LR records for common conditions.

#### Option B: Full PDF Extraction (Production)

```bash
# Ensure the McGee PDF is in the project root
python -m mcgee_app.data_ingestion
```

This will:
1. Extract appendix pages from the PDF
2. Convert pages to images
3. Use GPT-4o Vision to extract structured LR data
4. Load data into MongoDB

**Note:** Full extraction requires significant API usage and may take 30-60 minutes.

### 4. Run the Application

```bash
# From the project root
streamlit run mcgee_app/app.py

# Or from within mcgee_app directory
cd mcgee_app
streamlit run app.py
```

The application will open at `http://localhost:8501`

## 📊 Data Schema

Each maneuver record in MongoDB follows this schema:

```json
{
  "disease": "Deep Vein Thrombosis",
  "finding": "Calf swelling >3 cm compared to asymptomatic leg",
  "sensitivity": 0.56,
  "specificity": 0.74,
  "lr_positive": 2.1,
  "lr_negative": 0.6,
  "source_page": 652,
  "notes": "Measured 10 cm below tibial tuberosity",
  "_ingested_at": "2024-01-15T10:30:00Z",
  "_source": "mcgee_ebm_3rd_edition"
}
```

## 🎓 Understanding Likelihood Ratios

### What is a Likelihood Ratio?

A **Likelihood Ratio (LR)** tells you how much a test result changes the probability of disease:

- **LR+ (Positive LR)**: How much more likely disease is when the finding is PRESENT
- **LR- (Negative LR)**: How much more likely disease is when the finding is ABSENT

### Clinical Interpretation

| LR+ Value | Interpretation | Clinical Action |
|-----------|----------------|-----------------|
| > 10 | Large increase in probability | Strong evidence FOR disease |
| 5-10 | Moderate increase | Good evidence FOR disease |
| 2-5 | Small increase | Weak evidence |
| 1-2 | Minimal change | Finding not helpful |

| LR- Value | Interpretation | Clinical Action |
|-----------|----------------|-----------------|
| < 0.1 | Large decrease in probability | Strong evidence AGAINST disease |
| 0.1-0.2 | Moderate decrease | Good evidence AGAINST disease |
| 0.2-0.5 | Small decrease | Weak evidence |
| 0.5-1 | Minimal change | Finding not helpful |

### High-Yield Criteria

This application identifies maneuvers as **high-yield** if:
- **LR+ ≥ 5.0** - When positive, significantly increases disease probability
- **LR- ≤ 0.2** - When negative, significantly decreases disease probability

## 🔧 Configuration Options

Key settings in `config.py`:

```python
# LR Thresholds
HIGH_YIELD_LR_POSITIVE_THRESHOLD = 5.0
HIGH_YIELD_LR_NEGATIVE_THRESHOLD = 0.2
LOW_UTILITY_LR_LOWER = 0.5
LOW_UTILITY_LR_UPPER = 2.0

# Differential diagnosis count
MIN_DIFFERENTIAL_COUNT = 3
MAX_DIFFERENTIAL_COUNT = 5

# OpenAI settings
VISION_MODEL = "gpt-4o"
REASONING_MODEL = "gpt-4o"
TEMPERATURE_DIFFERENTIAL = 0.3
TEMPERATURE_SYNTHESIS = 0.5
```

## 🧪 Testing

### Test the RAG Engine

```bash
python -m mcgee_app.rag_engine
```

This runs a sample case through the pipeline.

### Test Configuration

```bash
python -m mcgee_app.config
```

Validates that all required configuration is present.

## 📁 File Structure

```
mcgee_app/
├── __init__.py           # Package initialization
├── config.py             # Configuration management
├── data_ingestion.py     # PDF to MongoDB pipeline
├── rag_engine.py         # Core RAG pipeline
├── app.py                # Streamlit web interface
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── temp_images/          # (Generated) Temporary image storage
```

## 🔒 Security Considerations

1. **Never commit API keys** - Use environment variables or `.env` files
2. **MongoDB authentication** - Use authenticated connections in production
3. **Input validation** - The application validates user input before processing
4. **Rate limiting** - Consider implementing rate limiting for production deployment

## 🚧 Production Deployment

For production deployment, consider:

1. **Use MongoDB Atlas** for managed database hosting
2. **Deploy on Streamlit Cloud** or containerize with Docker
3. **Implement authentication** for user access control
4. **Add logging and monitoring** for operational visibility
5. **Set up CI/CD** for automated testing and deployment

### Docker Deployment (Example)

```dockerfile
FROM python:3.11-slim

# Install Poppler
RUN apt-get update && apt-get install -y poppler-utils

WORKDIR /app
COPY mcgee_app/ ./mcgee_app/
COPY mcgee_app/requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "mcgee_app/app.py", "--server.address=0.0.0.0"]
```

## 📚 References

- McGee, S. (2018). Evidence-Based Physical Diagnosis (4th ed.). Elsevier.
- [Likelihood Ratios - CEBM](https://www.cebm.ox.ac.uk/resources/ebm-tools/likelihood-ratios)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [MongoDB Python Driver](https://pymongo.readthedocs.io/)

## 📄 License

This project is for educational purposes. The underlying textbook content is copyrighted by the publisher.

## 🤝 Contributing

Contributions are welcome! Please ensure any changes maintain IRB compliance and educational focus.



