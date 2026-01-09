"""
Data Ingestion Module for McGee EBM Physical Exam Strategy Application.

This module handles the one-time process of extracting structured Likelihood Ratio
data from the McGee PDF textbook and loading it into MongoDB.

Process Overview:
1. Extract appendix pages from PDF
2. Convert pages to high-resolution images
3. Use GPT-4o Vision to extract structured data from table images
4. Load structured data into MongoDB

Dependencies:
- pypdf: PDF manipulation
- pdf2image: PDF to image conversion (requires Poppler installation)
- PIL: Image processing
- openai: GPT-4o Vision API
- pymongo: MongoDB operations

IMPORTANT: Poppler must be installed separately:
- macOS: brew install poppler
- Ubuntu: sudo apt-get install poppler-utils
- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases
"""

import os
import sys
import json
import base64
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Third-party imports
try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    from PyPDF2 import PdfReader, PdfWriter

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("Warning: pdf2image not installed. Install with: pip install pdf2image")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Install with: pip install Pillow")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: pip install openai")

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    print("Warning: pymongo not installed. Install with: pip install pymongo")

# Local imports
from config import (
    OPENAI_API_KEY,
    MONGODB_URI,
    PDF_FILE_PATH,
    JSON_OUTPUT_PATH,
    TEMP_DIR,
    DATABASE_NAME,
    COLLECTION_NAME,
    APPENDIX_START_PAGE,
    APPENDIX_END_PAGE,
    PDF_DPI,
    VISION_MODEL,
    MAX_TOKENS_EXTRACTION,
    TEMPERATURE_EXTRACTION,
    MONGODB_INDEXES,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """Handles PDF isolation and page extraction."""
    
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.reader = None
        
    def load_pdf(self) -> bool:
        """Load the PDF file."""
        try:
            self.reader = PdfReader(str(self.pdf_path))
            logger.info(f"Loaded PDF with {len(self.reader.pages)} pages")
            return True
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            return False
    
    def get_total_pages(self) -> int:
        """Get total number of pages in PDF."""
        if self.reader:
            return len(self.reader.pages)
        return 0
    
    def extract_page_range(self, start: int, end: int, output_path: Path) -> bool:
        """
        Extract a range of pages to a new PDF file.
        
        Args:
            start: Starting page (0-indexed)
            end: Ending page (0-indexed, exclusive)
            output_path: Path for the output PDF
            
        Returns:
            bool: Success status
        """
        if not self.reader:
            logger.error("PDF not loaded. Call load_pdf() first.")
            return False
        
        try:
            writer = PdfWriter()
            total_pages = len(self.reader.pages)
            
            # Validate page range
            start = max(0, start)
            end = min(end, total_pages)
            
            for page_num in range(start, end):
                writer.add_page(self.reader.pages[page_num])
            
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
            
            logger.info(f"Extracted pages {start}-{end} to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract pages: {e}")
            return False


class ImageConverter:
    """Converts PDF pages to images for Vision API processing."""
    
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        
    def convert_pdf_to_images(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        """
        Convert all pages of a PDF to PNG images.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images
            
        Returns:
            List of paths to generated images
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("pdf2image is required for image conversion")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        
        try:
            logger.info(f"Converting PDF to images at {self.dpi} DPI...")
            images = convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                fmt='png'
            )
            
            for i, image in enumerate(images):
                image_path = output_dir / f"page_{i:04d}.png"
                image.save(str(image_path), 'PNG')
                image_paths.append(image_path)
                logger.debug(f"Saved: {image_path}")
            
            logger.info(f"Converted {len(images)} pages to images")
            return image_paths
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise


class VisionExtractor:
    """Uses GPT-4o Vision to extract structured data from table images."""
    
    # System prompt for structured data extraction
    EXTRACTION_PROMPT = """You are an expert medical data transcriber specializing in evidence-based medicine tables.

The attached image contains a table from the Appendix of McGee's Evidence-Based Physical Diagnosis textbook. This table contains Likelihood Ratio (LR) data for physical examination findings.

Your task is to carefully transcribe ALL data from the table into a structured JSON format.

For each row in the table, extract:
1. disease: The parent disease/diagnosis/clinical condition this finding relates to
2. finding: The physical examination maneuver or clinical finding
3. sensitivity: Sensitivity value (as a float between 0 and 1, or null if not provided)
4. specificity: Specificity value (as a float between 0 and 1, or null if not provided)  
5. lr_positive: Positive Likelihood Ratio (as a float, or null if not provided)
6. lr_negative: Negative Likelihood Ratio (as a float, or null if not provided)
7. source_page: The page number if visible, otherwise null
8. notes: Any additional notes or qualifiers mentioned (e.g., "pooled data", specific populations)

IMPORTANT RULES:
- Identify the PARENT DISEASE from table headers or section titles
- Convert percentages to decimals (e.g., 85% → 0.85)
- For LR values shown as ranges (e.g., "3.2-5.4"), use the midpoint
- If a cell shows "NS" (not significant) or "-", set the value to null
- If LR+ shows ">10" or "∞", use 99.0 as a placeholder
- If LR- shows "<0.1" or "≈0", use 0.01 as a placeholder
- Include ALL rows, even if some values are missing

Return ONLY a valid JSON array of objects. Do not include any explanation or markdown formatting.

Example output format:
[
  {
    "disease": "Deep Vein Thrombosis",
    "finding": "Homan's sign",
    "sensitivity": 0.33,
    "specificity": 0.67,
    "lr_positive": 1.0,
    "lr_negative": 1.0,
    "source_page": 652,
    "notes": null
  }
]"""

    def __init__(self, api_key: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library is required")
        self.client = OpenAI(api_key=api_key)
        
    def encode_image_to_base64(self, image_path: Path) -> str:
        """Encode an image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_from_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Extract structured data from a single table image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of extracted maneuver dictionaries
        """
        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            response = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": self.EXTRACTION_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please extract all Likelihood Ratio data from this table image."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=MAX_TOKENS_EXTRACTION,
                temperature=TEMPERATURE_EXTRACTION
            )
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Clean up the response (remove markdown if present)
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            data = json.loads(content)
            
            if not isinstance(data, list):
                data = [data]
            
            logger.info(f"Extracted {len(data)} records from {image_path.name}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {image_path.name}: {e}")
            logger.debug(f"Raw content: {content[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Failed to extract from {image_path.name}: {e}")
            return []
    
    def extract_from_all_images(self, image_paths: List[Path], 
                                 progress_callback=None) -> List[Dict[str, Any]]:
        """
        Extract data from all images.
        
        Args:
            image_paths: List of paths to image files
            progress_callback: Optional callback function(current, total)
            
        Returns:
            Combined list of all extracted records
        """
        all_data = []
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{total}: {image_path.name}")
            
            records = self.extract_from_image(image_path)
            all_data.extend(records)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        logger.info(f"Total records extracted: {len(all_data)}")
        return all_data


class MongoDBLoader:
    """Handles MongoDB operations for storing extracted data."""
    
    def __init__(self, uri: str, database_name: str, collection_name: str):
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo library is required")
        
        self.uri = uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        
    def connect(self) -> bool:
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.uri)
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            logger.info(f"Connected to MongoDB: {self.database_name}.{self.collection_name}")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def create_indexes(self) -> bool:
        """Create indexes for efficient querying."""
        try:
            for index_config in MONGODB_INDEXES:
                self.collection.create_index(
                    index_config["key"],
                    name=index_config["name"]
                )
                logger.info(f"Created index: {index_config['name']}")
            
            # Create text index for full-text search
            self.collection.create_index(
                [("disease", "text"), ("finding", "text")],
                name="text_search_index"
            )
            logger.info("Created text search index")
            
            return True
            
        except OperationFailure as e:
            logger.error(f"Failed to create indexes: {e}")
            return False
    
    def load_data(self, data: List[Dict[str, Any]], clear_existing: bool = True) -> int:
        """
        Load extracted data into MongoDB.
        
        Args:
            data: List of maneuver dictionaries
            clear_existing: Whether to clear existing data first
            
        Returns:
            Number of documents inserted
        """
        if self.collection is None:
            logger.error("Not connected to MongoDB")
            return 0
        
        try:
            if clear_existing:
                result = self.collection.delete_many({})
                logger.info(f"Cleared {result.deleted_count} existing documents")
            
            # Add metadata to each record
            for record in data:
                record["_ingested_at"] = datetime.utcnow()
                record["_source"] = "mcgee_ebm_3rd_edition"
            
            # Insert all records
            result = self.collection.insert_many(data)
            inserted_count = len(result.inserted_ids)
            
            logger.info(f"Inserted {inserted_count} documents into MongoDB")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return 0
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


def save_json_backup(data: List[Dict[str, Any]], output_path: Path) -> bool:
    """Save extracted data to JSON file as backup."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved JSON backup to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save JSON backup: {e}")
        return False


def run_full_ingestion_pipeline(
    skip_pdf_extraction: bool = False,
    skip_mongodb_load: bool = False,
    use_existing_json: bool = False
) -> Dict[str, Any]:
    """
    Run the complete data ingestion pipeline.
    
    Args:
        skip_pdf_extraction: Skip PDF processing, use existing images
        skip_mongodb_load: Skip MongoDB loading, only extract to JSON
        use_existing_json: Load from existing JSON instead of extracting
        
    Returns:
        Dictionary with pipeline results
    """
    results = {
        "success": False,
        "pages_extracted": 0,
        "images_created": 0,
        "records_extracted": 0,
        "records_loaded": 0,
        "errors": []
    }
    
    # Validate configuration
    if not OPENAI_API_KEY and not use_existing_json:
        results["errors"].append("OPENAI_API_KEY not configured")
        return results
    
    if not PDF_FILE_PATH.exists() and not use_existing_json:
        results["errors"].append(f"PDF file not found: {PDF_FILE_PATH}")
        return results
    
    all_data = []
    
    try:
        # Step 1: Use existing JSON or extract from PDF
        if use_existing_json and JSON_OUTPUT_PATH.exists():
            logger.info("Loading from existing JSON file...")
            with open(JSON_OUTPUT_PATH, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            results["records_extracted"] = len(all_data)
            
        else:
            # Step 1a: Extract appendix pages
            if not skip_pdf_extraction:
                logger.info("Step 1: Extracting appendix pages from PDF...")
                extractor = PDFExtractor(PDF_FILE_PATH)
                
                if not extractor.load_pdf():
                    results["errors"].append("Failed to load PDF")
                    return results
                
                total_pages = extractor.get_total_pages()
                logger.info(f"PDF has {total_pages} total pages")
                
                # Adjust page range if needed
                end_page = min(APPENDIX_END_PAGE, total_pages)
                
                appendix_pdf = TEMP_DIR / "appendix_only.pdf"
                if not extractor.extract_page_range(APPENDIX_START_PAGE, end_page, appendix_pdf):
                    results["errors"].append("Failed to extract appendix pages")
                    return results
                
                results["pages_extracted"] = end_page - APPENDIX_START_PAGE
                
                # Step 1b: Convert to images
                logger.info("Step 2: Converting pages to images...")
                converter = ImageConverter(dpi=PDF_DPI)
                image_paths = converter.convert_pdf_to_images(appendix_pdf, TEMP_DIR / "images")
                results["images_created"] = len(image_paths)
            else:
                # Use existing images
                image_dir = TEMP_DIR / "images"
                image_paths = sorted(image_dir.glob("*.png"))
                logger.info(f"Using {len(image_paths)} existing images")
            
            # Step 2: Extract data using Vision API
            logger.info("Step 3: Extracting data using GPT-4o Vision...")
            vision_extractor = VisionExtractor(OPENAI_API_KEY)
            all_data = vision_extractor.extract_from_all_images(image_paths)
            results["records_extracted"] = len(all_data)
            
            # Step 3: Save JSON backup
            save_json_backup(all_data, JSON_OUTPUT_PATH)
        
        # Step 4: Load into MongoDB
        if not skip_mongodb_load and all_data:
            logger.info("Step 4: Loading data into MongoDB...")
            loader = MongoDBLoader(MONGODB_URI, DATABASE_NAME, COLLECTION_NAME)
            
            if loader.connect():
                loader.create_indexes()
                results["records_loaded"] = loader.load_data(all_data)
                loader.close()
            else:
                results["errors"].append("Failed to connect to MongoDB")
        
        results["success"] = len(results["errors"]) == 0
        
    except Exception as e:
        logger.exception("Pipeline failed with exception")
        results["errors"].append(str(e))
    
    return results


def load_sample_data() -> List[Dict[str, Any]]:
    """
    Load sample LR data for testing when full extraction isn't available.
    
    This provides a representative sample of the type of data that would
    be extracted from the McGee textbook.
    """
    sample_data = [
        # Deep Vein Thrombosis
        {
            "disease": "Deep Vein Thrombosis",
            "finding": "Calf swelling >3 cm compared to asymptomatic leg",
            "sensitivity": 0.56,
            "specificity": 0.74,
            "lr_positive": 2.1,
            "lr_negative": 0.6,
            "source_page": 652,
            "notes": "Measured 10 cm below tibial tuberosity"
        },
        {
            "disease": "Deep Vein Thrombosis",
            "finding": "Pitting edema in symptomatic leg only",
            "sensitivity": 0.47,
            "specificity": 0.76,
            "lr_positive": 2.0,
            "lr_negative": 0.7,
            "source_page": 652,
            "notes": None
        },
        {
            "disease": "Deep Vein Thrombosis",
            "finding": "Homan's sign",
            "sensitivity": 0.33,
            "specificity": 0.67,
            "lr_positive": 1.0,
            "lr_negative": 1.0,
            "source_page": 652,
            "notes": "Low diagnostic utility"
        },
        {
            "disease": "Deep Vein Thrombosis",
            "finding": "Superficial venous dilation",
            "sensitivity": 0.24,
            "specificity": 0.95,
            "lr_positive": 4.8,
            "lr_negative": 0.8,
            "source_page": 652,
            "notes": None
        },
        # Pneumonia
        {
            "disease": "Pneumonia",
            "finding": "Fever (>37.8°C)",
            "sensitivity": 0.75,
            "specificity": 0.60,
            "lr_positive": 1.9,
            "lr_negative": 0.42,
            "source_page": 312,
            "notes": None
        },
        {
            "disease": "Pneumonia",
            "finding": "Egophony",
            "sensitivity": 0.15,
            "specificity": 0.96,
            "lr_positive": 4.1,
            "lr_negative": 0.9,
            "source_page": 312,
            "notes": "High specificity when present"
        },
        {
            "disease": "Pneumonia",
            "finding": "Bronchial breath sounds",
            "sensitivity": 0.14,
            "specificity": 0.96,
            "lr_positive": 3.5,
            "lr_negative": 0.9,
            "source_page": 312,
            "notes": None
        },
        {
            "disease": "Pneumonia",
            "finding": "Dullness to percussion",
            "sensitivity": 0.26,
            "specificity": 0.82,
            "lr_positive": 2.2,
            "lr_negative": 0.8,
            "source_page": 312,
            "notes": None
        },
        {
            "disease": "Pneumonia",
            "finding": "Crackles (rales)",
            "sensitivity": 0.51,
            "specificity": 0.75,
            "lr_positive": 2.0,
            "lr_negative": 0.65,
            "source_page": 312,
            "notes": None
        },
        # Heart Failure
        {
            "disease": "Heart Failure",
            "finding": "Third heart sound (S3 gallop)",
            "sensitivity": 0.13,
            "specificity": 0.99,
            "lr_positive": 11.0,
            "lr_negative": 0.88,
            "source_page": 418,
            "notes": "Highly specific when present"
        },
        {
            "disease": "Heart Failure",
            "finding": "Jugular venous distension",
            "sensitivity": 0.39,
            "specificity": 0.92,
            "lr_positive": 5.1,
            "lr_negative": 0.66,
            "source_page": 418,
            "notes": None
        },
        {
            "disease": "Heart Failure",
            "finding": "Hepatojugular reflux",
            "sensitivity": 0.24,
            "specificity": 0.94,
            "lr_positive": 4.0,
            "lr_negative": 0.8,
            "source_page": 418,
            "notes": None
        },
        {
            "disease": "Heart Failure",
            "finding": "Peripheral edema",
            "sensitivity": 0.53,
            "specificity": 0.72,
            "lr_positive": 1.9,
            "lr_negative": 0.65,
            "source_page": 418,
            "notes": "Low specificity"
        },
        {
            "disease": "Heart Failure",
            "finding": "Pulmonary crackles",
            "sensitivity": 0.60,
            "specificity": 0.78,
            "lr_positive": 2.8,
            "lr_negative": 0.51,
            "source_page": 418,
            "notes": None
        },
        # Appendicitis
        {
            "disease": "Appendicitis",
            "finding": "Right lower quadrant tenderness",
            "sensitivity": 0.81,
            "specificity": 0.53,
            "lr_positive": 1.7,
            "lr_negative": 0.36,
            "source_page": 485,
            "notes": None
        },
        {
            "disease": "Appendicitis",
            "finding": "Rebound tenderness",
            "sensitivity": 0.63,
            "specificity": 0.69,
            "lr_positive": 2.0,
            "lr_negative": 0.54,
            "source_page": 485,
            "notes": None
        },
        {
            "disease": "Appendicitis",
            "finding": "Psoas sign",
            "sensitivity": 0.16,
            "specificity": 0.95,
            "lr_positive": 3.2,
            "lr_negative": 0.88,
            "source_page": 485,
            "notes": "Suggests retrocecal appendix"
        },
        {
            "disease": "Appendicitis",
            "finding": "Obturator sign",
            "sensitivity": 0.08,
            "specificity": 0.94,
            "lr_positive": 1.3,
            "lr_negative": 0.98,
            "source_page": 485,
            "notes": "Low sensitivity limits utility"
        },
        {
            "disease": "Appendicitis",
            "finding": "Rovsing's sign",
            "sensitivity": 0.22,
            "specificity": 0.87,
            "lr_positive": 1.7,
            "lr_negative": 0.90,
            "source_page": 485,
            "notes": None
        },
        # Ascites
        {
            "disease": "Ascites",
            "finding": "Shifting dullness",
            "sensitivity": 0.83,
            "specificity": 0.56,
            "lr_positive": 1.9,
            "lr_negative": 0.3,
            "source_page": 498,
            "notes": "Requires >500 mL fluid"
        },
        {
            "disease": "Ascites",
            "finding": "Fluid wave",
            "sensitivity": 0.62,
            "specificity": 0.90,
            "lr_positive": 6.0,
            "lr_negative": 0.42,
            "source_page": 498,
            "notes": "Requires >1000 mL fluid"
        },
        {
            "disease": "Ascites",
            "finding": "Bulging flanks",
            "sensitivity": 0.78,
            "specificity": 0.44,
            "lr_positive": 1.4,
            "lr_negative": 0.5,
            "source_page": 498,
            "notes": "Low specificity"
        },
        {
            "disease": "Ascites",
            "finding": "Puddle sign",
            "sensitivity": 0.43,
            "specificity": 0.86,
            "lr_positive": 3.0,
            "lr_negative": 0.66,
            "source_page": 498,
            "notes": "Detects small amounts of ascites"
        },
        # Aortic Stenosis
        {
            "disease": "Aortic Stenosis",
            "finding": "Delayed carotid upstroke (parvus et tardus)",
            "sensitivity": 0.31,
            "specificity": 0.93,
            "lr_positive": 4.4,
            "lr_negative": 0.74,
            "source_page": 380,
            "notes": "More reliable in younger patients"
        },
        {
            "disease": "Aortic Stenosis",
            "finding": "Absent A2",
            "sensitivity": 0.64,
            "specificity": 0.98,
            "lr_positive": 32.0,
            "lr_negative": 0.37,
            "source_page": 380,
            "notes": "Highly specific for severe AS"
        },
        {
            "disease": "Aortic Stenosis",
            "finding": "Late peaking systolic murmur",
            "sensitivity": 0.67,
            "specificity": 0.89,
            "lr_positive": 6.0,
            "lr_negative": 0.37,
            "source_page": 380,
            "notes": "Suggests severe stenosis"
        },
        {
            "disease": "Aortic Stenosis",
            "finding": "Systolic murmur grade ≥3",
            "sensitivity": 0.80,
            "specificity": 0.75,
            "lr_positive": 3.2,
            "lr_negative": 0.27,
            "source_page": 380,
            "notes": None
        },
        # Hypothyroidism
        {
            "disease": "Hypothyroidism",
            "finding": "Slow ankle reflex relaxation",
            "sensitivity": 0.77,
            "specificity": 0.93,
            "lr_positive": 11.0,
            "lr_negative": 0.25,
            "source_page": 265,
            "notes": "Classic finding"
        },
        {
            "disease": "Hypothyroidism",
            "finding": "Coarse skin",
            "sensitivity": 0.60,
            "specificity": 0.81,
            "lr_positive": 3.2,
            "lr_negative": 0.49,
            "source_page": 265,
            "notes": None
        },
        {
            "disease": "Hypothyroidism",
            "finding": "Periorbital puffiness",
            "sensitivity": 0.60,
            "specificity": 0.90,
            "lr_positive": 6.0,
            "lr_negative": 0.44,
            "source_page": 265,
            "notes": None
        },
        {
            "disease": "Hypothyroidism",
            "finding": "Bradycardia",
            "sensitivity": 0.25,
            "specificity": 0.92,
            "lr_positive": 3.1,
            "lr_negative": 0.82,
            "source_page": 265,
            "notes": "HR <60 bpm"
        },
        # Pleural Effusion
        {
            "disease": "Pleural Effusion",
            "finding": "Asymmetric chest expansion",
            "sensitivity": 0.74,
            "specificity": 0.91,
            "lr_positive": 8.1,
            "lr_negative": 0.29,
            "source_page": 320,
            "notes": None
        },
        {
            "disease": "Pleural Effusion",
            "finding": "Dullness to percussion",
            "sensitivity": 0.89,
            "specificity": 0.81,
            "lr_positive": 4.7,
            "lr_negative": 0.14,
            "source_page": 320,
            "notes": "Highly sensitive"
        },
        {
            "disease": "Pleural Effusion",
            "finding": "Absent tactile fremitus",
            "sensitivity": 0.82,
            "specificity": 0.86,
            "lr_positive": 5.7,
            "lr_negative": 0.21,
            "source_page": 320,
            "notes": None
        },
        {
            "disease": "Pleural Effusion",
            "finding": "Decreased breath sounds",
            "sensitivity": 0.88,
            "specificity": 0.83,
            "lr_positive": 5.2,
            "lr_negative": 0.14,
            "source_page": 320,
            "notes": None
        },
        # Carpal Tunnel Syndrome
        {
            "disease": "Carpal Tunnel Syndrome",
            "finding": "Flick sign",
            "sensitivity": 0.93,
            "specificity": 0.96,
            "lr_positive": 21.4,
            "lr_negative": 0.07,
            "source_page": 564,
            "notes": "Highly accurate"
        },
        {
            "disease": "Carpal Tunnel Syndrome",
            "finding": "Classic/probable hand diagram",
            "sensitivity": 0.64,
            "specificity": 0.73,
            "lr_positive": 2.4,
            "lr_negative": 0.49,
            "source_page": 564,
            "notes": None
        },
        {
            "disease": "Carpal Tunnel Syndrome",
            "finding": "Tinel's sign",
            "sensitivity": 0.50,
            "specificity": 0.77,
            "lr_positive": 2.2,
            "lr_negative": 0.65,
            "source_page": 564,
            "notes": "Variable accuracy"
        },
        {
            "disease": "Carpal Tunnel Syndrome",
            "finding": "Phalen's sign",
            "sensitivity": 0.68,
            "specificity": 0.73,
            "lr_positive": 2.5,
            "lr_negative": 0.44,
            "source_page": 564,
            "notes": None
        },
        {
            "disease": "Carpal Tunnel Syndrome",
            "finding": "Thenar atrophy",
            "sensitivity": 0.18,
            "specificity": 0.97,
            "lr_positive": 5.4,
            "lr_negative": 0.85,
            "source_page": 564,
            "notes": "Suggests advanced disease"
        },
        # Rotator Cuff Tear
        {
            "disease": "Rotator Cuff Tear",
            "finding": "Weakness in external rotation",
            "sensitivity": 0.84,
            "specificity": 0.53,
            "lr_positive": 1.8,
            "lr_negative": 0.30,
            "source_page": 541,
            "notes": None
        },
        {
            "disease": "Rotator Cuff Tear",
            "finding": "Drop arm test positive",
            "sensitivity": 0.27,
            "specificity": 0.88,
            "lr_positive": 2.3,
            "lr_negative": 0.83,
            "source_page": 541,
            "notes": "Tests supraspinatus"
        },
        {
            "disease": "Rotator Cuff Tear",
            "finding": "Painful arc sign",
            "sensitivity": 0.75,
            "specificity": 0.67,
            "lr_positive": 2.3,
            "lr_negative": 0.37,
            "source_page": 541,
            "notes": "60-120 degrees abduction"
        },
        {
            "disease": "Rotator Cuff Tear",
            "finding": "Empty can test positive",
            "sensitivity": 0.69,
            "specificity": 0.62,
            "lr_positive": 1.8,
            "lr_negative": 0.50,
            "source_page": 541,
            "notes": "Tests supraspinatus"
        },
        # Meniscal Tear
        {
            "disease": "Meniscal Tear",
            "finding": "McMurray's test positive",
            "sensitivity": 0.53,
            "specificity": 0.85,
            "lr_positive": 3.5,
            "lr_negative": 0.55,
            "source_page": 586,
            "notes": "Most specific maneuver"
        },
        {
            "disease": "Meniscal Tear",
            "finding": "Joint line tenderness",
            "sensitivity": 0.79,
            "specificity": 0.43,
            "lr_positive": 1.4,
            "lr_negative": 0.49,
            "source_page": 586,
            "notes": "Sensitive but not specific"
        },
        {
            "disease": "Meniscal Tear",
            "finding": "Thessaly test positive",
            "sensitivity": 0.75,
            "specificity": 0.87,
            "lr_positive": 5.9,
            "lr_negative": 0.29,
            "source_page": 586,
            "notes": "Standing rotation test"
        },
        {
            "disease": "Meniscal Tear",
            "finding": "History of locking/catching",
            "sensitivity": 0.34,
            "specificity": 0.84,
            "lr_positive": 2.1,
            "lr_negative": 0.79,
            "source_page": 586,
            "notes": "Historical finding"
        },
    ]
    
    return sample_data


def load_sample_data_to_mongodb() -> Dict[str, Any]:
    """
    Load sample data into MongoDB for testing.
    
    Returns:
        Dictionary with load results
    """
    results = {
        "success": False,
        "records_loaded": 0,
        "errors": []
    }
    
    try:
        sample_data = load_sample_data()
        
        loader = MongoDBLoader(MONGODB_URI, DATABASE_NAME, COLLECTION_NAME)
        
        if loader.connect():
            loader.create_indexes()
            results["records_loaded"] = loader.load_data(sample_data)
            loader.close()
            results["success"] = True
        else:
            results["errors"].append("Failed to connect to MongoDB")
            
    except Exception as e:
        results["errors"].append(str(e))
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="McGee Data Ingestion Pipeline")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Load sample data instead of extracting from PDF"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Extract to JSON only, skip MongoDB"
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing JSON file instead of re-extracting"
    )
    
    args = parser.parse_args()
    
    if args.sample:
        print("Loading sample data to MongoDB...")
        results = load_sample_data_to_mongodb()
    else:
        print("Running full ingestion pipeline...")
        results = run_full_ingestion_pipeline(
            skip_mongodb_load=args.json_only,
            use_existing_json=args.use_existing
        )
    
    print("\nResults:")
    print(f"  Success: {results['success']}")
    print(f"  Records loaded: {results.get('records_loaded', 'N/A')}")
    if results.get('errors'):
        print(f"  Errors: {results['errors']}")



