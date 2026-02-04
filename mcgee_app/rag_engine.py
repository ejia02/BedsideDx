"""
RAG Engine for McGee EBM Physical Exam Strategy Application.

This module implements the core RAG (Retrieval-Augmented Generation) pipeline:
1. Generate differential diagnosis from patient symptoms using GPT
2. Retrieve relevant physical exam evidence from MongoDB
3. Synthesize educational physical exam strategy

Exports:
- MongoDBClient: Database connection and query handling
- run_rag_pipeline: Main RAG pipeline function
- run_rag_pipeline_with_sample_data: Fallback using sample data
- synthesize_exam_strategy_structured: Structured output by body system
- parse_strategy_json: Parse GPT JSON responses
- PYMONGO_AVAILABLE, OPENAI_AVAILABLE: Dependency flags
"""

# IMPORTANT: Set environment variables BEFORE importing torch/sentence_transformers
# This prevents MPS (Apple Silicon GPU) from being used, avoiding meta tensor issues
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for pymongo availability
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
    import certifi
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    logger.warning("pymongo not installed. Install with: pip install pymongo certifi")

# Check for openai availability
try:
    import openai as openai_module
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    OPENAI_VERSION = getattr(openai_module, "__version__", "unknown")
except ImportError:
    OPENAI_AVAILABLE = False
    OPENAI_VERSION = "not-installed"
    logger.warning("openai not installed. Install with: pip install openai")

# Check for sentence_transformers availability (for vector search)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence_transformers not installed. Install with: pip install sentence-transformers")

# Import configuration
from config import (
    OPENAI_API_KEY,
    MONGODB_URI,
    DATABASE_NAME,
    COLLECTION_NAME,
    REASONING_MODEL,
    REASONING_MODEL_SUPPORTS_TEMPERATURE,
    MAX_TOKENS_DIFFERENTIAL,
    MAX_TOKENS_SYNTHESIS,
    TEMPERATURE_DIFFERENTIAL,
    TEMPERATURE_SYNTHESIS,
    MIN_DIFFERENTIAL_COUNT,
    MAX_DIFFERENTIAL_COUNT,
    HIGH_YIELD_LR_POSITIVE_THRESHOLD,
    HIGH_YIELD_LR_NEGATIVE_THRESHOLD,
    LOW_UTILITY_LR_LOWER,
    LOW_UTILITY_LR_UPPER,
)

# Standard body systems for physical exam documentation
# These match the canonical format used in clinical PE writeups
STANDARD_BODY_SYSTEMS = [
    "Constitutional/General",
    "Eyes",
    "HEENT",
    "Cardiovascular",
    "Respiratory",
    "Gastrointestinal",
    "Genitourinary",
    "Musculoskeletal",
    "Integumentary",
    "Neurological",
    "Endocrine",
    "Hematological/Lymphatic",
    "Allergic/Immunological",
    "Psychiatric",
]

# Mapping of keywords to body systems for categorization guidance
BODY_SYSTEM_KEYWORDS = {
    "Constitutional/General": ["vital signs", "temperature", "blood pressure", "heart rate", "respiratory rate", "BMI", "general appearance", "gait", "posture", "weight", "height", "nutrition"],
    "Eyes": ["ophthalmologic", "fundoscopic", "pupils", "vision", "visual acuity", "eye exam", "retina", "optic disc", "sclera", "conjunctiva"],
    "HEENT": ["head", "ears", "nose", "throat", "mouth", "oral", "tympanic", "hearing", "sinuses", "pharynx", "tonsils", "neck", "thyroid", "carotid", "JVP", "jugular", "trachea", "cervical"],
    "Cardiovascular": ["heart", "cardiac", "murmur", "S1", "S2", "S3", "S4", "PMI", "pulse", "edema", "aortic", "mitral", "tricuspid", "peripheral pulses", "capillary refill"],
    "Respiratory": ["lung", "breath sounds", "wheeze", "crackles", "percussion chest", "fremitus", "egophony", "rhonchi", "rales", "respiratory effort"],
    "Gastrointestinal": ["abdomen", "liver", "spleen", "bowel", "ascites", "hepatomegaly", "splenomegaly", "tenderness abdom", "rebound", "guarding", "rectal", "hernia"],
    "Genitourinary": ["genitalia", "urinary", "prostate", "testicular", "vaginal", "pelvic", "CVA tenderness", "bladder"],
    "Musculoskeletal": ["joint", "range of motion", "strength", "muscle", "bone", "extremity", "ankle", "knee", "hip", "shoulder", "wrist", "hand exam", "foot exam", "back", "spine"],
    "Integumentary": ["skin", "rash", "lesion", "wound", "ulcer", "turgor", "dermatologic", "breast", "nail", "hair"],
    "Neurological": ["reflex", "cranial nerve", "sensory", "motor", "cerebellar", "mental status", "coordination", "Babinski", "clonus", "tone", "tremor", "gait neuro"],
    "Endocrine": ["thyroid enlargement", "goiter", "exophthalmos", "acanthosis", "hirsutism", "gynecomastia"],
    "Hematological/Lymphatic": ["lymph node", "lymphadenopathy", "axillary", "inguinal", "supraclavicular", "splenomegaly", "petechiae", "purpura", "ecchymosis"],
    "Allergic/Immunological": ["urticaria", "angioedema", "anaphylaxis signs", "allergic reaction", "immune response"],
    "Psychiatric": ["mood", "affect", "thought process", "thought content", "judgment", "insight", "orientation", "memory", "behavior"],
}


class MongoDBClient:
    """
    MongoDB client for physical exam evidence database.
    
    Handles connection management, querying by disease/finding,
    and text search operations.
    """
    
    def __init__(self, uri: str = None, database_name: str = None, collection_name: str = None):
        """
        Initialize MongoDB client.
        
        Args:
            uri: MongoDB connection string (defaults to config value)
            database_name: Database name (defaults to config value)
            collection_name: Collection name (defaults to config value)
        """
        self.uri = uri or MONGODB_URI
        self.database_name = database_name or DATABASE_NAME
        self.collection_name = collection_name or COLLECTION_NAME
        self.client = None
        self.db = None
        self.collection = None
        self._connected = False
        
        # Initialize embedding model for vector search (free local model)
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Loading SentenceTransformer model for vector search...")
                # Environment variables set at module level to force CPU usage
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                logger.info("SentenceTransformer model loaded successfully on CPU")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer model: {e}")
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not PYMONGO_AVAILABLE:
            logger.error("pymongo is not installed")
            return False
        
        if not self.uri:
            logger.error("MongoDB URI not configured")
            return False
        
        try:
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                tlsCAFile=certifi.where()
            )
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            self._connected = True
            
            logger.info(f"Connected to MongoDB: {self.database_name}.{self.collection_name}")
            
            # Ensure indexes exist
            self._ensure_indexes()
            self._check_embedding_fields()
            
            return True
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            logger.error(f"MongoDB connection error: {e}")
            return False
    
    def _ensure_indexes(self) -> None:
        """Ensure required indexes exist on the collection."""
        try:
            existing_indexes = self.collection.index_information()
            
            # Create text search index if it doesn't exist
            # Uses fields from exam_evidence_free nested schema
            if "text_search_index" not in existing_indexes:
                logger.info("Creating text search index...")
                self.collection.create_index(
                    [("source.ebm_box_label", "text"), ("original_finding", "text"), ("text_for_embedding", "text")],
                    name="text_search_index"
                )
                logger.info("Text search index created")
            
            # Create single field indexes
            # Field names match exam_evidence_free nested schema
            for field in ["source.ebm_box_label", "original_finding", "result_buckets.lr_positive", "result_buckets.lr_negative"]:
                index_name = f"{field}_index"
                if index_name not in existing_indexes:
                    self.collection.create_index(field, name=index_name)
                    logger.info(f"Created index: {index_name}")
                    
        except OperationFailure as e:
            logger.warning(f"Could not create indexes (may already exist): {e}")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")

    def _check_embedding_fields(self) -> None:
        """Warn if expected embedding fields are missing."""
        try:
            if not self.collection.find_one({"ebm_box_label_embedding": {"$exists": True}}, {"_id": 1}):
                logger.warning(
                    "No ebm_box_label_embedding field found. "
                    "Disease vector search will fall back to regex. "
                    "Run migrate_ebm_label_embeddings.py or reingest."
                )
            if not self.collection.find_one({"embedding": {"$exists": True}}, {"_id": 1}):
                logger.warning(
                    "No embedding field found. Vector search will fall back to text/regex search."
                )
        except Exception as e:
            logger.warning(f"Could not verify embedding fields: {e}")
    
    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("MongoDB connection closed")
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self.collection is not None
    
    def get_all_diseases(self) -> List[str]:
        """
        Get list of all unique diseases/diagnoses (source.ebm_box_label) in the collection.
        
        Returns:
            List of disease/diagnosis names from source.ebm_box_label field
        """
        if not self.is_connected():
            return []
        
        try:
            return self.collection.distinct("source.ebm_box_label")
        except Exception as e:
            logger.error(f"Failed to get diseases: {e}")
            return []
    
    def get_maneuvers_by_disease(self, disease: str) -> List[Dict[str, Any]]:
        """
        Get all maneuvers/findings for a specific disease/diagnosis.
        
        Args:
            disease: Disease/diagnosis name to search for (matches source.ebm_box_label)
            
        Returns:
            List of maneuver documents
        """
        if not self.is_connected():
            return []
        
        try:
            # Case-insensitive regex search on source.ebm_box_label field
            results = list(self.collection.find(
                {"source.ebm_box_label": {"$regex": disease, "$options": "i"}},
                {"_id": 0, "embedding": 0}  # Exclude these fields
            ))
            return results
        except Exception as e:
            logger.error(f"Failed to query maneuvers for {disease}: {e}")
            return []
    
    def get_maneuvers_by_diseases(self, diseases: List[str]) -> List[Dict[str, Any]]:
        """
        Get maneuvers for multiple diseases using disease-focused vector similarity search.
        
        Uses semantic similarity search on ebm_box_label embeddings to find relevant 
        maneuvers based on the disease names. This enables matching abbreviations and
        synonyms (e.g., "DVT" -> "Deep Vein Thrombosis", "CHF" -> "Heart Failure").
        
        Falls back to regex search if vector search fails.
        
        Args:
            diseases: List of disease names from differential diagnosis
            
        Returns:
            List of maneuver documents sorted by relevance
        """
        if not self.is_connected() or not diseases:
            return []
        
        all_maneuvers = []
        seen_ids = set()
        
        # Use disease-focused vector search for each disease in the differential
        if self.embedding_model:
            for disease in diseases:
                # Search using ebm_box_label embeddings for semantic disease matching
                results = self.disease_vector_search(disease, limit=20)
                
                # Deduplicate results based on document content
                for result in results:
                    # Create a unique key for deduplication
                    doc_key = f"{result.get('source', {}).get('ebm_box_label', '')}:{result.get('original_finding', '')}"
                    if doc_key not in seen_ids:
                        seen_ids.add(doc_key)
                        all_maneuvers.append(result)
            
            if all_maneuvers:
                logger.info(f"Disease vector search found {len(all_maneuvers)} unique maneuvers for diseases: {diseases}")
                return all_maneuvers
        
        # Fallback to regex-based search for each disease
        logger.info("Using regex fallback search for diseases")
        for disease in diseases:
            maneuvers = self.get_maneuvers_by_disease(disease)
            for maneuver in maneuvers:
                doc_key = f"{maneuver.get('source', {}).get('ebm_box_label', '')}:{maneuver.get('original_finding', '')}"
                if doc_key not in seen_ids:
                    seen_ids.add(doc_key)
                    all_maneuvers.append(maneuver)
        
        logger.info(f"Regex search found {len(all_maneuvers)} unique maneuvers for diseases: {diseases}")
        return all_maneuvers
    
    def text_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Perform text search on disease and finding fields.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        if not self.is_connected():
            return []
        
        try:
            results = list(self.collection.find(
                {"$text": {"$search": query}},
                {"_id": 0, "embedding": 0, "score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit))
            
            return results
        except OperationFailure as e:
            logger.error(f"Text search failed: {e.details.get('errmsg', str(e))}, full error: {e.details}")
            # Fallback to regex search
            return self._fallback_search(query, limit)
        except Exception as e:
            logger.error(f"Text search error: {e}")
            return self._fallback_search(query, limit)
    
    def _fallback_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fallback regex-based search when text search is unavailable.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            # Split query into terms
            terms = query.split()
            
            # Build OR conditions for each term
            # Search across nested schema fields
            or_conditions = []
            for term in terms:
                or_conditions.extend([
                    {"source.ebm_box_label": {"$regex": term, "$options": "i"}},
                    {"maneuver.name": {"$regex": term, "$options": "i"}},
                    {"original_finding": {"$regex": term, "$options": "i"}},
                    {"text_for_embedding": {"$regex": term, "$options": "i"}}
                ])
            
            if not or_conditions:
                return []
            
            results = list(self.collection.find(
                {"$or": or_conditions},
                {"_id": 0, "embedding": 0}
            ).limit(limit))
            
            return results
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    def vector_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search using vector embeddings.
        
        Uses the SentenceTransformer model to generate query embeddings and
        MongoDB Atlas $vectorSearch to find similar documents.
        
        Args:
            query: Search query string (will be converted to embedding)
            limit: Maximum number of results
            
        Returns:
            List of matching documents sorted by similarity score
        """
        if not self.is_connected():
            return []
        
        if not self.embedding_model:
            logger.warning("Embedding model not available, falling back to text search")
            return self._fallback_search(query, limit)
        
        try:
            # Generate query embedding using local SentenceTransformer model
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Build MongoDB Atlas vector search aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index_free",  # Index for 384-dim embeddings
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$project": {
                        "embedding": 0  # Exclude embedding from results
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Vector search returned {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except OperationFailure as e:
            error_msg = str(e).lower()
            if "index not found" in error_msg or "vector" in error_msg:
                logger.warning(f"Vector search index not available: {e}")
                logger.info("Falling back to text search. To enable vector search, create a vector index in MongoDB Atlas.")
            else:
                logger.error(f"Vector search failed: {e}")
            return self._fallback_search(query, limit)
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return self._fallback_search(query, limit)
    
    def disease_vector_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Perform disease-focused vector search using ebm_box_label embeddings.
        
        This searches against the ebm_box_label_embedding field which contains
        embeddings of disease/diagnosis names. Useful for finding documents
        when you have a disease name or differential diagnosis.
        
        Enables semantic matching like:
        - "DVT" -> "Deep Vein Thrombosis"
        - "CHF" -> "Heart Failure"  
        - "MI" -> "Myocardial Infarction"
        
        Args:
            query: Disease name or related term to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents sorted by similarity score
        """
        if not self.is_connected():
            return []
        
        if not self.embedding_model:
            logger.warning("Embedding model not available, falling back to regex search")
            return self._fallback_disease_search(query, limit)
        
        try:
            # Generate query embedding using local SentenceTransformer model
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Build MongoDB Atlas vector search aggregation pipeline for disease-focused search
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "ebm_label_vector_index",  # Disease-focused index
                        "path": "ebm_box_label_embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$project": {
                        "embedding": 0,  # Exclude main embedding from results
                        "ebm_box_label_embedding": 0  # Exclude label embedding too
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Disease vector search returned {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except OperationFailure as e:
            error_msg = str(e).lower()
            if "index not found" in error_msg or "vector" in error_msg:
                logger.warning(f"Disease vector search index not available: {e}")
                logger.info("Falling back to regex search. To enable disease vector search, create ebm_label_vector_index in MongoDB Atlas.")
            else:
                logger.error(f"Disease vector search failed: {e}")
            return self._fallback_disease_search(query, limit)
        except Exception as e:
            logger.error(f"Disease vector search error: {e}")
            return self._fallback_disease_search(query, limit)
    
    def _fallback_disease_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fallback regex-based search for disease names when vector search is unavailable.
        
        Args:
            query: Disease name to search for
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        if not self.is_connected():
            return []
        
        try:
            results = list(self.collection.find(
                {"source.ebm_box_label": {"$regex": query, "$options": "i"}},
                {"_id": 0, "embedding": 0, "ebm_box_label_embedding": 0}
            ).limit(limit))
            
            return results
        except Exception as e:
            logger.error(f"Fallback disease search failed: {e}")
            return []


def _extract_lr_values(item: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract LR+ and LR- values from evidence item, handling nested schema.
    
    The MongoDB schema stores LR values in result_buckets[0].lr_positive/lr_negative,
    but we also support flat schema with pos_lr_numeric/neg_lr_numeric for backwards compatibility.
    
    Args:
        item: Evidence document from MongoDB
        
    Returns:
        Tuple of (lr_positive, lr_negative) as floats or None
    """
    # Try nested schema first (result_buckets)
    result_buckets = item.get("result_buckets", [])
    if result_buckets:
        bucket = result_buckets[0] if result_buckets else {}
        lr_pos = bucket.get("lr_positive")
        lr_neg = bucket.get("lr_negative")
        if lr_pos is not None or lr_neg is not None:
            return lr_pos, lr_neg
    
    # Fallback to flat schema
    return item.get("pos_lr_numeric"), item.get("neg_lr_numeric")


def _extract_diagnosis_and_finding(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract diagnosis (ebm_box_label) and finding from evidence item, handling nested schema.
    
    The MongoDB schema stores diagnosis in source.ebm_box_label,
    but we also support flat schema for backwards compatibility.
    
    Args:
        item: Evidence document from MongoDB
        
    Returns:
        Tuple of (diagnosis, finding) as strings
    """
    # Try nested schema first
    source = item.get("source", {})
    diagnosis = source.get("ebm_box_label") or item.get("ebm_box_label", "Unknown")
    finding = item.get("original_finding") or item.get("finding_text", "Unknown")
    
    return diagnosis, finding


def _extract_maneuver_base(item: Dict[str, Any]) -> str:
    """
    Extract the base maneuver name from evidence item.
    
    The maneuver_base field contains the core exam maneuver (e.g., "Auscultation of heart")
    which helps determine the body system for categorization.
    
    Args:
        item: Evidence document from MongoDB
        
    Returns:
        Maneuver base name as string
    """
    # Try nested schema first (maneuver.name)
    maneuver = item.get("maneuver", {})
    maneuver_base = maneuver.get("name") or item.get("maneuver_base", "")
    
    # If still empty, try to extract from original_finding
    if not maneuver_base:
        maneuver_base = item.get("original_finding", "Unknown maneuver")
    
    return maneuver_base


def categorize_evidence(evidence: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize evidence by likelihood ratio utility.
    
    Categories:
    - high_yield_positive: LR+ >= threshold (strongly rules in)
    - high_yield_negative: LR- <= threshold (strongly rules out)
    - low_utility: LR between 0.5 and 2.0 (doesn't change probability much)
    - other: Everything else
    
    Handles nested MongoDB schema where LR values are in result_buckets[0].
    
    Args:
        evidence: List of evidence documents from MongoDB
        
    Returns:
        Dictionary with categorized evidence lists
    """
    categories = {
        "high_yield_positive": [],
        "high_yield_negative": [],
        "low_utility": [],
        "other": []
    }
    
    missing_lr_count = 0
    for item in evidence:
        # Extract LR values from nested or flat schema
        lr_pos, lr_neg = _extract_lr_values(item)
        if lr_pos is None and lr_neg is None:
            missing_lr_count += 1
        
        categorized = False
        
        # Check for high yield positive LR
        if lr_pos is not None and lr_pos >= HIGH_YIELD_LR_POSITIVE_THRESHOLD:
            categories["high_yield_positive"].append(item)
            categorized = True
        
        # Check for high yield negative LR (can be in multiple categories)
        if lr_neg is not None and lr_neg <= HIGH_YIELD_LR_NEGATIVE_THRESHOLD:
            categories["high_yield_negative"].append(item)
            categorized = True
        
        # Check for low utility (only if not already high yield)
        if not categorized:
            is_low_utility_pos = (lr_pos is not None and 
                                  LOW_UTILITY_LR_LOWER <= lr_pos <= LOW_UTILITY_LR_UPPER)
            is_low_utility_neg = (lr_neg is not None and 
                                  LOW_UTILITY_LR_LOWER <= lr_neg <= LOW_UTILITY_LR_UPPER)
            
            if is_low_utility_pos or is_low_utility_neg:
                categories["low_utility"].append(item)
                categorized = True
        
        # Everything else
        if not categorized:
            categories["other"].append(item)
    
    if missing_lr_count:
        logger.info(
            "Evidence items missing LR values: %s/%s",
            missing_lr_count,
            len(evidence)
        )
    
    return categories


def _get_structured_strategy_schema() -> Dict[str, Any]:
    """Return the JSON schema for structured exam strategy output."""
    return {
        "name": "structured_exam_strategy",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "system": {"type": "string"},
                            "diseases": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "rule_in": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "lr_positive": {
                                                        "type": ["string", "number", "null"]
                                                    },
                                                    "technique": {"type": ["string", "null"]}
                                                },
                                                "required": ["name", "lr_positive", "technique"],
                                                "additionalProperties": False
                                            }
                                        },
                                        "rule_out": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "lr_negative": {
                                                        "type": ["string", "number", "null"]
                                                    },
                                                    "technique": {"type": ["string", "null"]}
                                                },
                                                "required": ["name", "lr_negative", "technique"],
                                                "additionalProperties": False
                                            }
                                        }
                                    },
                                    "required": ["name", "rule_in", "rule_out"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["system", "diseases"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["sections"],
            "additionalProperties": False
        }
    }


def _extract_json_from_response(response: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract JSON payload from a Responses or Chat Completions response."""
    try:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return json.loads(output_text), None
    except Exception as exc:
        return None, f"Failed to parse output_text JSON: {exc}"

    try:
        outputs = getattr(response, "output", None)
        if outputs:
            for output in outputs:
                contents = getattr(output, "content", None) or []
                for content in contents:
                    if isinstance(content, dict):
                        if "json" in content and content["json"] is not None:
                            return content["json"], None
                        if "text" in content and content["text"]:
                            return json.loads(content["text"]), None
                    json_payload = getattr(content, "json", None)
                    if json_payload is not None:
                        return json_payload, None
                    text_payload = getattr(content, "text", None)
                    if text_payload:
                        return json.loads(text_payload), None
    except Exception as exc:
        return None, f"Failed to parse response output JSON: {exc}"

    try:
        choices = getattr(response, "choices", None)
        if choices:
            content = choices[0].message.content
            if content:
                return json.loads(content), None
    except Exception as exc:
        return None, f"Failed to parse choices JSON: {exc}"

    return None, "No JSON payload found in response"


def _parse_differential_items(content: str) -> Optional[List[Dict[str, str]]]:
    """Parse JSON differential items from model output."""
    if not content:
        return None

    payload = None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                payload = None

    if payload is None:
        return None

    items = None
    if isinstance(payload, dict):
        items = payload.get("differential")
    elif isinstance(payload, list):
        items = payload

    if not isinstance(items, list):
        return None

    parsed_items: List[Dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            name = item.strip()
            rationale = ""
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("diagnosis") or item.get("dx") or "").strip()
            rationale = str(item.get("rationale") or item.get("reason") or "").strip()
        else:
            continue
        if name:
            parsed_items.append({"name": name, "rationale": rationale})

    return parsed_items or None


def _normalize_differential_items(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Normalize differential items with deduplication and length limits."""
    normalized: List[Dict[str, str]] = []
    seen = set()
    for item in items:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append({
            "name": name,
            "rationale": (item.get("rationale") or "").strip()
        })
        if len(normalized) >= MAX_DIFFERENTIAL_COUNT:
            break
    return normalized


def _request_structured_strategy_json(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str
) -> Tuple[Optional[Union[Dict[str, Any], str]], Optional[str]]:
    """Request a structured exam strategy using JSON schema enforcement."""
    schema_payload = _get_structured_strategy_schema()
    try:
        logger.info("OpenAI SDK version: %s", OPENAI_VERSION)
        logger.info("Requesting structured output via Responses API (json_schema, strict mode).")
        response = client.responses.create(
            model=REASONING_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": schema_payload
            },
            max_output_tokens=MAX_TOKENS_SYNTHESIS,
            **({"temperature": TEMPERATURE_SYNTHESIS} if REASONING_MODEL_SUPPORTS_TEMPERATURE else {})
        )
        payload, error = _extract_json_from_response(response)
        if error:
            raise ValueError(error)
        try:
            payload_text = json.dumps(payload, ensure_ascii=True)
            preview = payload_text[:1200]
            logger.info(
                "Responses JSON payload size=%s preview=%s",
                len(payload_text),
                preview
            )
        except Exception as exc:
            logger.warning("Failed to serialize Responses payload for logging: %s", exc)
        return payload, None
    except TypeError as exc:
        if "response_format" not in str(exc):
            logger.warning(f"Responses API JSON schema failed, falling back: {exc}")
        else:
            logger.warning(
                "Responses API response_format unsupported, trying text.format fallback: %s",
                exc
            )
            try:
                response = client.responses.create(
                    model=REASONING_MODEL,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "json_schema": schema_payload
                        }
                    },
                    max_output_tokens=MAX_TOKENS_SYNTHESIS,
                    **({"temperature": TEMPERATURE_SYNTHESIS} if REASONING_MODEL_SUPPORTS_TEMPERATURE else {})
                )
                payload, error = _extract_json_from_response(response)
                if error:
                    raise ValueError(error)
                try:
                    payload_text = json.dumps(payload, ensure_ascii=True)
                    preview = payload_text[:1200]
                    logger.info(
                        "Responses text.format payload size=%s preview=%s",
                        len(payload_text),
                        preview
                    )
                except Exception as log_exc:
                    logger.warning("Failed to serialize text.format payload for logging: %s", log_exc)
                return payload, None
            except Exception as fallback_exc:
                logger.warning(f"Responses API text.format failed, falling back: {fallback_exc}")
    except Exception as exc:
        logger.warning(f"Responses API JSON schema failed, falling back: {exc}")

    try:
        logger.info("Requesting structured output via chat.completions fallback (json_schema).")
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=MAX_TOKENS_SYNTHESIS,
            response_format={
                "type": "json_schema",
                "json_schema": schema_payload
            },
            **({"temperature": TEMPERATURE_SYNTHESIS} if REASONING_MODEL_SUPPORTS_TEMPERATURE else {})
        )
        message = response.choices[0].message
        
        # Check for refusal first
        refusal = getattr(message, 'refusal', None)
        if refusal:
            logger.warning(f"Model refused to generate response: {refusal}")
            return None, f"Model refused: {refusal}"
        
        # Try multiple possible locations for the JSON content
        content = None
        
        # 1. Try message.parsed (structured output parsed JSON)
        parsed = getattr(message, 'parsed', None)
        if parsed is not None:
            logger.info("Found parsed JSON in message.parsed")
            if isinstance(parsed, dict):
                return parsed, None
            try:
                content = json.dumps(parsed)
            except (TypeError, ValueError):
                content = str(parsed)
        
        # 2. Try message.content (standard location)
        if not content:
            content = message.content
        
        if not content:
            # Log available message attributes for debugging
            msg_attrs = [attr for attr in dir(message) if not attr.startswith('_')]
            logger.error(f"Empty response from model. Message attributes: {msg_attrs}")
            return None, "Empty response from model"
        
        preview = content[:1200]
        logger.info(
            "Chat JSON payload size=%s preview=%s",
            len(content),
            preview
        )
        return content, None
    except Exception as exc:
        return None, str(exc)


def generate_differential_diagnosis(symptoms: str, client: OpenAI) -> Dict[str, Any]:
    """
    Generate differential diagnosis from patient symptoms using GPT.

    Args:
        symptoms: Patient symptom description
        status_callback: Optional function called with stage updates
        client: OpenAI client instance

    Returns:
        Dictionary with:
        - diagnoses: List of diagnosis names
        - details: List of {name, rationale} items
    """
    prompt = f"""You are an experienced physician educator. Based on the following patient presentation,
generate a focused differential diagnosis list of {MIN_DIFFERENTIAL_COUNT}-{MAX_DIFFERENTIAL_COUNT} conditions
that should be considered.

Patient Presentation:
{symptoms}

Instructions:
1. Include both common and serious (don't miss) diagnoses
2. Provide 1-2 concise sentences of clinical rationale per diagnosis
3. Return ONLY JSON with this schema:
{{
  "differential": [
    {{"name": "Diagnosis name", "rationale": "1-2 sentence rationale"}}
  ]
}}
4. Use standard medical terminology
5. Do not include numbering or extra keys

Your differential diagnosis JSON:"""

    try:
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a physician educator helping generate differential diagnoses for educational purposes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=MAX_TOKENS_DIFFERENTIAL,
            **({"temperature": TEMPERATURE_DIFFERENTIAL} if REASONING_MODEL_SUPPORTS_TEMPERATURE else {})
        )
        
        content = response.choices[0].message.content
        if not content:
            logger.error("Model returned empty response for differential diagnosis")
            return {"diagnoses": [], "details": []}
        content = content.strip()

        parsed_items = _parse_differential_items(content)
        if parsed_items:
            normalized_items = _normalize_differential_items(parsed_items)
        else:
            diagnoses = [dx.strip() for dx in content.split(",") if dx.strip()]
            diagnoses = diagnoses[:MAX_DIFFERENTIAL_COUNT]
            normalized_items = [{"name": dx, "rationale": ""} for dx in diagnoses]

        diagnoses = [item["name"] for item in normalized_items]
        logger.info(f"Generated {len(diagnoses)} differential diagnoses")
        return {"diagnoses": diagnoses, "details": normalized_items}
        
    except Exception as e:
        logger.error(f"Failed to generate differential diagnosis: {e}")
        return {"diagnoses": [], "details": []}


def synthesize_exam_strategy_structured(
    symptoms: str,
    differential: List[str],
    evidence: List[Dict[str, Any]],
    categories: Dict[str, List[Dict[str, Any]]],
    client: OpenAI
) -> Dict[str, Any]:
    """
    Synthesize an educational physical exam strategy as structured data.
    
    Returns JSON-structured output organized by body system, then by disease,
    with maneuvers categorized as "Rule In" (LR+ ≥ 2.0) or "Rule Out" (LR- ≤ 0.5).
    
    Args:
        symptoms: Original patient symptoms
        differential: List of differential diagnoses
        evidence: Retrieved evidence from database
        categories: Categorized evidence by LR utility
        client: OpenAI client instance
        
    Returns:
        Dictionary with 'sections' list containing body system exams with disease groupings
    """
    logger.info(
        "Structured strategy input: differential_count=%s evidence_count=%s",
        len(differential),
        len(evidence)
    )

    # Build evidence summary organized by disease, extracting from nested schema
    evidence_by_disease = {}
    
    # Process all evidence and group by disease
    for item in evidence:
        diagnosis, finding = _extract_diagnosis_and_finding(item)
        maneuver_base = _extract_maneuver_base(item)
        lr_pos, lr_neg = _extract_lr_values(item)
        
        if diagnosis not in evidence_by_disease:
            evidence_by_disease[diagnosis] = {"rule_in": [], "rule_out": [], "other": []}
        
        finding_info = {
            "finding": finding,
            "maneuver": maneuver_base,
            "lr_positive": lr_pos,
            "lr_negative": lr_neg
        }
        
        # Categorize based on LR values (LR+ ≥ 2.0 for rule in, LR- ≤ 0.5 for rule out)
        if lr_pos is not None and lr_pos >= 2.0:
            evidence_by_disease[diagnosis]["rule_in"].append(finding_info)
        if lr_neg is not None and lr_neg <= 0.5:
            evidence_by_disease[diagnosis]["rule_out"].append(finding_info)
        if (lr_pos is None or lr_pos < 2.0) and (lr_neg is None or lr_neg > 0.5):
            evidence_by_disease[diagnosis]["other"].append(finding_info)
    
    # Build formatted evidence summary with maneuver names for body system inference
    evidence_summary = ""
    for disease, findings in evidence_by_disease.items():
        if findings["rule_in"] or findings["rule_out"]:
            evidence_summary += f"\n**{disease}:**\n"
            if findings["rule_in"]:
                evidence_summary += "  Rule In (LR+ ≥ 2.0):\n"
                for f in findings["rule_in"][:5]:
                    lr_str = f"{f['lr_positive']:.1f}" if f['lr_positive'] else "N/A"
                    maneuver_str = f"[{f['maneuver']}] " if f['maneuver'] else ""
                    evidence_summary += f"    - {maneuver_str}{f['finding']}: LR+ = {lr_str}\n"
            if findings["rule_out"]:
                evidence_summary += "  Rule Out (LR- ≤ 0.5):\n"
                for f in findings["rule_out"][:5]:
                    lr_str = f"{f['lr_negative']:.2f}" if f['lr_negative'] else "N/A"
                    maneuver_str = f"[{f['maneuver']}] " if f['maneuver'] else ""
                    evidence_summary += f"    - {maneuver_str}{f['finding']}: LR- = {lr_str}\n"
    
    if not evidence_summary:
        evidence_summary = "No specific evidence available in database for these conditions."
    logger.info("Evidence summary length=%s characters", len(evidence_summary))
    
    # Build body systems list for prompt
    body_systems_list = "\n".join(f"  - {system}" for system in STANDARD_BODY_SYSTEMS)
    
    prompt = f"""Create a focused physical exam strategy for this patient.

Patient presentation:
{symptoms}

Differential diagnosis:
{', '.join(differential)}

Available evidence (McGee):
{evidence_summary}

BODY SYSTEMS FORMAT:
Organize findings using ONLY these standard physical exam body systems (in this order):
{body_systems_list}

Use the exact system names above for the "system" field. Categorize maneuvers as follows:
- Constitutional/General: vital signs, temperature, BP, HR, RR, general appearance, gait, posture
- Eyes: ophthalmologic exam, fundoscopy, pupils, visual acuity, sclera, conjunctiva
- HEENT: head, ears, nose, mouth, throat, neck, thyroid, carotid, JVP, trachea
- Cardiovascular: heart sounds, murmurs, PMI, pulses, edema, capillary refill
- Respiratory: breath sounds, lung percussion, wheeze, crackles, fremitus, respiratory effort
- Gastrointestinal: abdominal exam, liver, spleen, bowel sounds, tenderness, ascites, rectal
- Genitourinary: genitalia exam, urinary, prostate, pelvic, CVA tenderness
- Musculoskeletal: joints, ROM, strength, back/spine, extremity exams
- Integumentary: skin exam, rashes, lesions, wounds, turgor, breast exam, nails
- Neurological: reflexes, cranial nerves, sensory, motor, cerebellar, mental status
- Endocrine: thyroid enlargement, goiter, acanthosis, hirsutism
- Hematological/Lymphatic: lymph nodes, lymphadenopathy, splenomegaly, petechiae
- Allergic/Immunological: urticaria, angioedema, allergic reaction signs
- Psychiatric: mood, affect, thought process, judgment, insight, orientation

Rules:
- Rule In: LR+ ≥ 2.0. Rule Out: LR- ≤ 0.5.
- Only include maneuvers relevant to this case; prioritize high LR values.
- Limit to 3-4 maneuvers per disease and omit low-probability diagnoses.
- A disease can appear under multiple body systems.
- Only include body systems that have relevant findings for this case (omit empty sections)."""

    system_prompt = (
        "You are an expert physician educator specializing in evidence-based physical diagnosis. "
        "Select only clinically useful maneuvers for this specific presentation. "
        "Return valid JSON that matches the provided schema."
    )

    try:
        payload, error = _request_structured_strategy_json(
            client=client,
            system_prompt=system_prompt,
            user_prompt=prompt
        )
        if error:
            logger.error(f"Model returned error for structured strategy: {error}")
            return {"sections": [], "error": error}

        structured_strategy = parse_strategy_json(payload)
        if structured_strategy.get("parse_error"):
            logger.error(f"Structured strategy parse error: {structured_strategy['parse_error']}")

        if not structured_strategy.get("sections"):
            logger.warning(
                "Structured strategy returned 0 sections. differential_count=%s evidence_count=%s",
                len(differential),
                len(evidence)
            )

        logger.info(
            "Generated structured exam strategy with %s sections",
            len(structured_strategy.get("sections", []))
        )
        return structured_strategy
        
    except Exception as e:
        logger.error(f"Failed to synthesize structured strategy: {e}")
        return {"sections": [], "error": str(e)}


def parse_strategy_json(content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse the JSON strategy response from GPT.
    
    Handles the new hierarchical structure:
    sections -> diseases -> rule_in/rule_out -> maneuvers
    
    Args:
        content: Raw response content from GPT
        
    Returns:
        Parsed dictionary with sections list containing disease groupings
    """
    def normalize_lr_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return str(value)
        return str(value).strip()

    if isinstance(content, dict):
        result: Dict[str, Any] = content
    else:
        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        content = content.strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strategy JSON: {e}")
            logger.debug(f"Raw content: {content[:500]}")
            return {"sections": [], "parse_error": str(e)}

    if not isinstance(result, dict):
        return {"sections": [], "parse_error": "Strategy JSON is not an object"}

    # Validate structure
    if "sections" not in result or not isinstance(result.get("sections"), list):
        result["sections"] = []

    # Ensure each section has required fields
    for section in result.get("sections", []):
        if not isinstance(section, dict):
            continue
        if "system" not in section or not section.get("system"):
            section["system"] = "General Exam"

        # Ensure diseases array exists
        if "diseases" not in section or not isinstance(section.get("diseases"), list):
            section["diseases"] = []

        # Validate each disease entry
        for disease in section.get("diseases", []):
            if not isinstance(disease, dict):
                continue
            disease["name"] = str(disease.get("name") or "Unknown Disease")
            disease.setdefault("rule_in", [])
            disease.setdefault("rule_out", [])
            if not isinstance(disease.get("rule_in"), list):
                disease["rule_in"] = []
            if not isinstance(disease.get("rule_out"), list):
                disease["rule_out"] = []

            # Validate rule_in maneuvers
            for maneuver in disease.get("rule_in", []):
                if not isinstance(maneuver, dict):
                    continue
                maneuver["name"] = str(maneuver.get("name") or "Unknown maneuver")
                maneuver["lr_positive"] = normalize_lr_value(maneuver.get("lr_positive", ""))
                maneuver["technique"] = str(maneuver.get("technique") or "")

            # Validate rule_out maneuvers
            for maneuver in disease.get("rule_out", []):
                if not isinstance(maneuver, dict):
                    continue
                maneuver["name"] = str(maneuver.get("name") or "Unknown maneuver")
                maneuver["lr_negative"] = normalize_lr_value(maneuver.get("lr_negative", ""))
                maneuver["technique"] = str(maneuver.get("technique") or "")

        # Backwards compatibility: convert old "maneuvers" format to new structure
        if "maneuvers" in section and section["maneuvers"] and not section["diseases"]:
            general_disease = {
                "name": "General Assessment",
                "rule_in": [],
                "rule_out": []
            }
            for maneuver in section["maneuvers"]:
                if not isinstance(maneuver, dict):
                    continue
                lr_info = str(maneuver.get("lr_info", ""))
                if "LR+" in lr_info or maneuver.get("high_yield", False):
                    general_disease["rule_in"].append({
                        "name": maneuver.get("name", "Unknown"),
                        "lr_positive": lr_info.replace("LR+ ", "").replace("LR- ", ""),
                        "technique": maneuver.get("technique", "")
                    })
                elif "LR-" in lr_info:
                    general_disease["rule_out"].append({
                        "name": maneuver.get("name", "Unknown"),
                        "lr_negative": lr_info.replace("LR- ", "").replace("LR+ ", ""),
                        "technique": maneuver.get("technique", "")
                    })
            if general_disease["rule_in"] or general_disease["rule_out"]:
                section["diseases"] = [general_disease]

    # Remove diseases with no maneuvers and drop empty sections
    pruned_sections = []
    for section in result.get("sections", []):
        diseases = section.get("diseases", [])
        if diseases:
            section["diseases"] = [
                disease for disease in diseases
                if isinstance(disease, dict) and (disease.get("rule_in") or disease.get("rule_out"))
            ]
        if section.get("diseases"):
            pruned_sections.append(section)
    
    result["sections"] = pruned_sections

    return result


def synthesize_exam_strategy(
    symptoms: str,
    differential: List[str],
    evidence: List[Dict[str, Any]],
    categories: Dict[str, List[Dict[str, Any]]],
    client: OpenAI
) -> str:
    """
    Legacy function - synthesize strategy as markdown text.
    
    Kept for backwards compatibility. New code should use
    synthesize_exam_strategy_structured() instead.
    """
    # Build evidence summary, extracting from nested schema
    evidence_summary = ""
    
    if categories["high_yield_positive"]:
        evidence_summary += "\n**High-Yield Positive Findings (LR+ ≥ 10.0):**\n"
        for item in categories["high_yield_positive"][:10]:
            diagnosis, finding = _extract_diagnosis_and_finding(item)
            lr_pos, _ = _extract_lr_values(item)
            lr_str = f"{lr_pos:.1f}" if lr_pos is not None else "N/A"
            evidence_summary += f"- {finding} for {diagnosis}: LR+ = {lr_str}\n"
    
    if categories["high_yield_negative"]:
        evidence_summary += "\n**High-Yield Negative Findings (LR- ≤ 0.1):**\n"
        for item in categories["high_yield_negative"][:10]:
            diagnosis, finding = _extract_diagnosis_and_finding(item)
            _, lr_neg = _extract_lr_values(item)
            lr_str = f"{lr_neg:.2f}" if lr_neg is not None else "N/A"
            evidence_summary += f"- {finding} for {diagnosis}: LR- = {lr_str}\n"
    
    if not evidence_summary:
        evidence_summary = "No specific evidence available in database for these conditions."
    
    prompt = f"""You are a physician educator creating an evidence-based physical exam teaching strategy.

**Patient Presentation:**
{symptoms}

**Differential Diagnosis:**
{', '.join(differential)}

**Available Evidence:**
{evidence_summary}

Create a concise physical exam strategy grouped by body system. Focus on high-yield maneuvers.
Format in clear markdown. Include LR values where available."""

    try:
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert physician educator specializing in evidence-based physical diagnosis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=MAX_TOKENS_SYNTHESIS,
            **({"temperature": TEMPERATURE_SYNTHESIS} if REASONING_MODEL_SUPPORTS_TEMPERATURE else {})
        )
        
        content = response.choices[0].message.content
        if not content:
            logger.error("Model returned empty response for exam strategy")
            return "Unable to generate strategy due to empty model response."
        strategy = content.strip()
        logger.info("Generated exam strategy successfully")
        return strategy
        
    except Exception as e:
        logger.error(f"Failed to synthesize strategy: {e}")
        return "Unable to generate strategy. Please try again."


def run_rag_pipeline(
    symptoms: str,
    status_callback: Optional[Callable[[str], None]] = None,
    differential_callback: Optional[Callable[[List[str], List[Dict[str, str]]], None]] = None
) -> Dict[str, Any]:
    """
    Run the complete RAG pipeline for physical exam strategy generation.
    
    Pipeline Steps:
    1. Generate differential diagnosis from symptoms
    2. Retrieve relevant evidence from MongoDB
    3. Categorize evidence by LR utility
    4. Synthesize educational exam strategy (structured by body system)
    
    Args:
        symptoms: Patient symptom description
        status_callback: Optional function called with stage updates
        differential_callback: Optional function called immediately when differential
            is generated, before evidence retrieval. Receives (diagnoses, details).
        
    Returns:
        Dictionary containing:
        - success: bool
        - differential: List of diagnoses
        - differential_details: List of diagnoses with rationales
        - evidence: List of evidence documents
        - categories: Categorized evidence
        - strategy: Legacy markdown strategy (deprecated)
        - strategy_structured: Structured strategy with body system sections
        - processing_time: Time taken in seconds
        - error: Error message if failed
    """
    start_time = time.time()
    
    result = {
        "success": False,
        "differential": [],
        "evidence": [],
        "categories": {},
        "strategy": "",
        "strategy_structured": {"sections": []},
        "processing_time": 0,
        "error": None,
        "differential_details": []
    }

    def _status(message: str):
        if status_callback:
            status_callback(message)
    
    # Validate dependencies
    if not OPENAI_AVAILABLE:
        result["error"] = "OpenAI library not installed"
        return result
    
    if not OPENAI_API_KEY:
        result["error"] = "OpenAI API key not configured"
        return result
    
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Step 1: Generate differential diagnosis
        logger.info("Step 1: Generating differential diagnosis...")
        _status("Generating differential diagnosis...")
        differential_payload = generate_differential_diagnosis(symptoms, openai_client)
        differential = differential_payload.get("diagnoses", [])
        differential_details = differential_payload.get("details", [])

        if not differential:
            result["error"] = "Failed to generate differential diagnosis"
            return result
        
        result["differential"] = differential
        result["differential_details"] = differential_details
        
        # Immediately notify about differential so UI can display it early
        if differential_callback:
            differential_callback(differential, differential_details)
        
        # Step 2: Retrieve evidence from MongoDB
        logger.info("Step 2: Retrieving evidence from MongoDB...")
        _status("Retrieving evidence...")
        mongo_client = MongoDBClient()
        
        evidence = []
        if mongo_client.connect():
            # Get maneuvers for each diagnosis
            evidence = mongo_client.get_maneuvers_by_diseases(differential)
            
            # Also do text search for symptoms
            symptom_evidence = mongo_client.text_search(symptoms, limit=10)
            
            # Merge and deduplicate using nested schema fields (source.ebm_box_label)
            combined = []
            seen_findings = set()
            for item in evidence + symptom_evidence:
                diagnosis, finding = _extract_diagnosis_and_finding(item)
                finding_key = f"{diagnosis}:{finding}"
                if finding_key not in seen_findings:
                    seen_findings.add(finding_key)
                    combined.append(item)
            
            evidence = combined
            
            mongo_client.close()
        
        logger.info(f"Retrieved {len(evidence)} maneuvers for {len(differential)} diseases")
        result["evidence"] = evidence
        
        if not evidence:
            logger.warning("No evidence retrieved from MongoDB. Database may be empty.")
        
        # Step 3: Categorize evidence
        _status("Categorizing evidence...")
        categories = categorize_evidence(evidence)
        result["categories"] = categories
        
        # Step 4: Synthesize structured strategy (by body system)
        logger.info("Step 4: Synthesizing structured exam strategy...")
        _status("Synthesizing exam strategy...")
        strategy_structured = synthesize_exam_strategy_structured(
            symptoms, differential, evidence, categories, openai_client
        )
        result["strategy_structured"] = strategy_structured
        
        # Also generate legacy markdown for backwards compatibility
        strategy = synthesize_exam_strategy(
            symptoms, differential, evidence, categories, openai_client
        )
        result["strategy"] = strategy
        
        result["success"] = True
        
    except Exception as e:
        logger.exception("Pipeline error")
        result["error"] = str(e)
    
    _status("Finalizing response...")
    result["processing_time"] = time.time() - start_time
    logger.info(f"Pipeline completed in {result['processing_time']:.2f} seconds")
    
    return result


def run_rag_pipeline_with_sample_data(
    symptoms: str,
    status_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Run RAG pipeline using sample data (fallback when MongoDB unavailable).
    
    This function uses the sample data from data_ingestion.py instead of
    querying MongoDB, useful for testing or when database is unavailable.
    
    Args:
        symptoms: Patient symptom description
        
    Returns:
        Same structure as run_rag_pipeline()
    """
    start_time = time.time()
    
    result = {
        "success": False,
        "differential": [],
        "evidence": [],
        "categories": {},
        "strategy": "",
        "strategy_structured": {"sections": []},
        "processing_time": 0,
        "error": None,
        "using_sample_data": True,
        "differential_details": []
    }

    def _status(message: str):
        if status_callback:
            status_callback(message)
    
    # Validate dependencies
    if not OPENAI_AVAILABLE:
        result["error"] = "OpenAI library not installed"
        return result
    
    if not OPENAI_API_KEY:
        result["error"] = "OpenAI API key not configured"
        return result
    
    try:
        # Import sample data
        from data_ingestion import load_sample_data
        sample_data = load_sample_data()
        
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Step 1: Generate differential diagnosis
        logger.info("Step 1: Generating differential diagnosis...")
        _status("Generating differential diagnosis...")
        differential_payload = generate_differential_diagnosis(symptoms, openai_client)
        differential = differential_payload.get("diagnoses", [])
        differential_details = differential_payload.get("details", [])

        if not differential:
            result["error"] = "Failed to generate differential diagnosis"
            return result
        
        result["differential"] = differential
        result["differential_details"] = differential_details
        
        # Step 2: Filter sample data by differential
        logger.info("Step 2: Filtering sample data by differential...")
        _status("Retrieving evidence...")
        evidence = []
        for item in sample_data:
            disease = item.get("disease", "").lower()
            for dx in differential:
                if dx.lower() in disease or disease in dx.lower():
                    evidence.append(item)
                    break
        
        logger.info(f"Found {len(evidence)} relevant maneuvers in sample data")
        result["evidence"] = evidence
        
        # Step 3: Categorize evidence
        _status("Categorizing evidence...")
        categories = categorize_evidence(evidence)
        result["categories"] = categories
        
        # Step 4: Synthesize structured strategy
        logger.info("Step 4: Synthesizing structured exam strategy...")
        _status("Synthesizing exam strategy...")
        strategy_structured = synthesize_exam_strategy_structured(
            symptoms, differential, evidence, categories, openai_client
        )
        result["strategy_structured"] = strategy_structured
        
        # Also generate legacy markdown
        strategy = synthesize_exam_strategy(
            symptoms, differential, evidence, categories, openai_client
        )
        result["strategy"] = strategy
        
        result["success"] = True
        
    except Exception as e:
        logger.exception("Pipeline error (sample data mode)")
        result["error"] = str(e)
    
    _status("Finalizing response...")
    result["processing_time"] = time.time() - start_time
    logger.info(f"Pipeline (sample data) completed in {result['processing_time']:.2f} seconds")
    
    return result


# For testing
if __name__ == "__main__":
    # Test MongoDB connection
    print("Testing MongoDB connection...")
    client = MongoDBClient()
    
    if client.connect():
        print(f"✅ Connected successfully")
        diseases = client.get_all_diseases()
        print(f"📋 Found {len(diseases)} diseases in database")
        
        if diseases:
            print(f"   Sample diseases: {diseases[:5]}")
        
        # Test search
        results = client.text_search("heart failure")
        print(f"🔍 Text search for 'heart failure': {len(results)} results")
        
        client.close()
    else:
        print("❌ Connection failed")
    
    # Test pipeline with sample symptoms
    print("\n" + "="*50)
    print("Testing RAG pipeline...")
    
    test_symptoms = """
    45-year-old woman with left leg swelling and pain for 3 days. 
    Redness and warmth in calf. Recent 12-hour flight one week ago.
    """
    
    result = run_rag_pipeline(test_symptoms)
    
    if result["success"]:
        print(f"✅ Pipeline succeeded in {result['processing_time']:.2f}s")
        print(f"📋 Differential: {result['differential']}")
        print(f"📚 Evidence items: {len(result['evidence'])}")
        print(f"🎯 High-yield positive: {len(result['categories'].get('high_yield_positive', []))}")
        print(f"🎯 High-yield negative: {len(result['categories'].get('high_yield_negative', []))}")
    else:
        print(f"❌ Pipeline failed: {result['error']}")
