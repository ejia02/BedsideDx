"""
Configuration settings for the McGee EBM Physical Exam Strategy Application.

This module contains all configuration variables, API keys, and constants
required for the application to function.

IMPORTANT: Never commit actual API keys or connection strings to version control.
Use environment variables or a .env file for sensitive credentials.
"""

import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Find and load .env file from project root
BASE_DIR = Path(__file__).parent.parent
ENV_FILE = BASE_DIR / ".env"

# Load .env file if it exists
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    print(f"✅ Loaded .env from: {ENV_FILE}")
else:
    # Also try current working directory
    load_dotenv()

# =============================================================================
# API KEYS AND CONNECTION STRINGS
# =============================================================================
# These should be set as environment variables for security
# Example: export OPENAI_API_KEY="your-key-here"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MONGODB_URI = os.getenv("MONGODB_URI", "")

# =============================================================================
# FILE PATHS AND NAMES
# =============================================================================

# BASE_DIR already defined above for .env loading

# PDF source file
PDF_FILE_NAME = "mcgee-evidence-based-physical-diagnosis-3rd-ed1.pdf"
PDF_FILE_PATH = BASE_DIR / PDF_FILE_NAME

# Output files for data processing
JSON_OUTPUT_NAME = "mcgee_data_raw.json"
JSON_OUTPUT_PATH = BASE_DIR / "mcgee_app" / JSON_OUTPUT_NAME

# Temporary directory for image processing
TEMP_DIR = BASE_DIR / "mcgee_app" / "temp_images"

# =============================================================================
# MONGODB CONFIGURATION
# =============================================================================

# Read from environment or use defaults
DATABASE_NAME = os.getenv("DATABASE_NAME", "bedside_dx")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "exam_evidence_free")

# Index configuration for efficient queries
# Field names match the exam_evidence_free collection schema from excel_parser.py
MONGODB_INDEXES = [
    {"key": "ebm_box_label", "name": "ebm_box_label_index"},
    {"key": "original_finding", "name": "original_finding_index"},
    {"key": "pos_lr_numeric", "name": "pos_lr_numeric_index"},
    {"key": "neg_lr_numeric", "name": "neg_lr_numeric_index"},
]

# =============================================================================
# PDF PROCESSING CONFIGURATION
# =============================================================================

# Appendix page range (0-indexed for pypdf)
# The Appendix with LR tables typically starts around page 650-700
# Adjust these values after manual inspection of the PDF
APPENDIX_START_PAGE = 649  # Page 650 in 1-indexed
APPENDIX_END_PAGE = 700    # Adjust based on actual appendix length

# Image conversion settings for pdf2image
PDF_DPI = 300  # Higher DPI for better OCR/Vision accuracy
IMAGE_FORMAT = "PNG"

# =============================================================================
# OPENAI API CONFIGURATION
# =============================================================================

# Model settings
VISION_MODEL = "gpt-4o"  # For PDF image extraction
REASONING_MODEL = "gpt-4o"  # For differential diagnosis and synthesis

# Token limits
MAX_TOKENS_EXTRACTION = 4096
MAX_TOKENS_DIFFERENTIAL = 1024
MAX_TOKENS_SYNTHESIS = 2048

# Temperature settings (lower = more deterministic)
TEMPERATURE_EXTRACTION = 0.1  # Very low for accurate data extraction
TEMPERATURE_DIFFERENTIAL = 0.3  # Slightly higher for clinical reasoning
TEMPERATURE_SYNTHESIS = 0.5  # Moderate for educational content

# =============================================================================
# RAG PIPELINE CONFIGURATION
# =============================================================================

# Number of differential diagnoses to generate
MIN_DIFFERENTIAL_COUNT = 3
MAX_DIFFERENTIAL_COUNT = 5

# LR thresholds for categorizing maneuvers
HIGH_YIELD_LR_POSITIVE_THRESHOLD = 5.0  # LR+ > 5.0 is high yield
HIGH_YIELD_LR_NEGATIVE_THRESHOLD = 0.2  # LR- < 0.2 is high yield
LOW_UTILITY_LR_LOWER = 0.5  # LR between 0.5 and 2.0 is low utility
LOW_UTILITY_LR_UPPER = 2.0

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

APP_TITLE = "EBM Physical Exam Strategist (Educational Tool)"
APP_VERSION = "1.0.0"

# IRB Compliance warning message
IRB_WARNING = """
⚠️ **IMPORTANT DISCLAIMER**

This tool is for **educational purposes only** and must not replace clinical 
judgment or official diagnosis. The information provided is intended to support 
medical education and should be used as a learning aid alongside proper clinical 
training and supervision.

Always consult with qualified healthcare professionals for actual patient care.
"""

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def validate_config() -> dict:
    """
    Validate that all required configuration is present.
    
    Returns:
        dict: Status of configuration validation with any errors.
    """
    errors = []
    warnings = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set. Set it via environment variable.")
    
    if MONGODB_URI == "mongodb://localhost:27017":
        warnings.append("Using default MongoDB URI. Set MONGODB_URI for production.")
    
    if not PDF_FILE_PATH.exists():
        errors.append(f"PDF file not found at: {PDF_FILE_PATH}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


if __name__ == "__main__":
    # Quick validation check
    result = validate_config()
    print("Configuration Validation:")
    print(f"  Valid: {result['valid']}")
    if result['errors']:
        print(f"  Errors: {result['errors']}")
    if result['warnings']:
        print(f"  Warnings: {result['warnings']}")



