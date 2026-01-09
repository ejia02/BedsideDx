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

import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for pymongo availability
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    logger.warning("pymongo not installed. Install with: pip install pymongo")

# Check for openai availability
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
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
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("SentenceTransformer model loaded successfully")
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
                connectTimeoutMS=10000
            )
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            self._connected = True
            
            logger.info(f"Connected to MongoDB: {self.database_name}.{self.collection_name}")
            
            # Ensure indexes exist
            self._ensure_indexes()
            
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
            # Uses fields from exam_evidence_free schema (excel_parser.py)
            if "text_search_index" not in existing_indexes:
                logger.info("Creating text search index...")
                self.collection.create_index(
                    [("ebm_box_label", "text"), ("original_finding", "text"), ("text_for_embedding", "text")],
                    name="text_search_index"
                )
                logger.info("Text search index created")
            
            # Create single field indexes
            # Field names match exam_evidence_free collection schema
            for field in ["ebm_box_label", "original_finding", "pos_lr_numeric", "neg_lr_numeric"]:
                index_name = f"{field}_index"
                if index_name not in existing_indexes:
                    self.collection.create_index(field, name=index_name)
                    logger.info(f"Created index: {index_name}")
                    
        except OperationFailure as e:
            logger.warning(f"Could not create indexes (may already exist): {e}")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
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
        Get list of all unique diseases/diagnoses (ebm_box_label) in the collection.
        
        Returns:
            List of disease/diagnosis names from ebm_box_label field
        """
        if not self.is_connected():
            return []
        
        try:
            return self.collection.distinct("ebm_box_label")
        except Exception as e:
            logger.error(f"Failed to get diseases: {e}")
            return []
    
    def get_maneuvers_by_disease(self, disease: str) -> List[Dict[str, Any]]:
        """
        Get all maneuvers/findings for a specific disease/diagnosis.
        
        Args:
            disease: Disease/diagnosis name to search for (matches ebm_box_label)
            
        Returns:
            List of maneuver documents
        """
        if not self.is_connected():
            return []
        
        try:
            # Case-insensitive regex search on ebm_box_label field
            results = list(self.collection.find(
                {"ebm_box_label": {"$regex": disease, "$options": "i"}},
                {"_id": 0, "embedding": 0}  # Exclude these fields
            ))
            return results
        except Exception as e:
            logger.error(f"Failed to query maneuvers for {disease}: {e}")
            return []
    
    def get_maneuvers_by_diseases(self, diseases: List[str]) -> List[Dict[str, Any]]:
        """
        Get maneuvers for multiple diseases using vector similarity search.
        
        Uses semantic similarity search to find relevant maneuvers based on
        the disease names. Falls back to regex search if vector search fails.
        
        Args:
            diseases: List of disease names from differential diagnosis
            
        Returns:
            List of maneuver documents sorted by relevance
        """
        if not self.is_connected() or not diseases:
            return []
        
        # Build a combined query from all diseases for vector search
        query = " ".join(diseases)
        
        # Try vector search first (semantic similarity)
        if self.embedding_model:
            results = self.vector_search(query, limit=50)
            if results:
                logger.info(f"Vector search found {len(results)} maneuvers for diseases: {diseases}")
                return results
        
        # Fallback to regex-based search for each disease
        logger.info("Using regex fallback search for diseases")
        all_maneuvers = []
        for disease in diseases:
            maneuvers = self.get_maneuvers_by_disease(disease)
            all_maneuvers.extend(maneuvers)
        
        logger.info(f"Regex search found {len(all_maneuvers)} maneuvers for diseases: {diseases}")
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
            # Search across ebm_box_label, original_finding, and text_for_embedding
            or_conditions = []
            for term in terms:
                or_conditions.extend([
                    {"ebm_box_label": {"$regex": term, "$options": "i"}},
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


def categorize_evidence(evidence: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize evidence by likelihood ratio utility.
    
    Categories:
    - high_yield_positive: LR+ >= threshold (strongly rules in)
    - high_yield_negative: LR- <= threshold (strongly rules out)
    - low_utility: LR between 0.5 and 2.0 (doesn't change probability much)
    - other: Everything else
    
    Args:
        evidence: List of evidence documents with pos_lr_numeric/neg_lr_numeric fields
        
    Returns:
        Dictionary with categorized evidence lists
    """
    categories = {
        "high_yield_positive": [],
        "high_yield_negative": [],
        "low_utility": [],
        "other": []
    }
    
    for item in evidence:
        # Use field names from exam_evidence_free schema
        lr_pos = item.get("pos_lr_numeric")
        lr_neg = item.get("neg_lr_numeric")
        
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
    
    return categories


def generate_differential_diagnosis(symptoms: str, client: OpenAI) -> List[str]:
    """
    Generate differential diagnosis from patient symptoms using GPT.
    
    Args:
        symptoms: Patient symptom description
        client: OpenAI client instance
        
    Returns:
        List of differential diagnoses
    """
    prompt = f"""You are an experienced physician educator. Based on the following patient presentation, 
generate a focused differential diagnosis list of {MIN_DIFFERENTIAL_COUNT}-{MAX_DIFFERENTIAL_COUNT} conditions 
that should be considered.

Patient Presentation:
{symptoms}

Instructions:
1. List the most likely diagnoses based on the presentation
2. Include both common and serious (don't miss) diagnoses
3. Return ONLY a comma-separated list of diagnosis names
4. Use standard medical terminology
5. Do not include explanations or numbering

Example output format:
Deep Vein Thrombosis, Cellulitis, Baker's Cyst, Muscle Strain

Your differential diagnosis:"""

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
            max_tokens=MAX_TOKENS_DIFFERENTIAL,
            temperature=TEMPERATURE_DIFFERENTIAL
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse comma-separated list
        diagnoses = [dx.strip() for dx in content.split(",") if dx.strip()]
        
        # Limit to max count
        diagnoses = diagnoses[:MAX_DIFFERENTIAL_COUNT]
        
        logger.info(f"Generated {len(diagnoses)} differential diagnoses")
        return diagnoses
        
    except Exception as e:
        logger.error(f"Failed to generate differential diagnosis: {e}")
        return []


def synthesize_exam_strategy_structured(
    symptoms: str,
    differential: List[str],
    evidence: List[Dict[str, Any]],
    categories: Dict[str, List[Dict[str, Any]]],
    client: OpenAI
) -> Dict[str, Any]:
    """
    Synthesize an educational physical exam strategy as structured data.
    
    Returns JSON-structured output organized by body system for
    rendering as collapsible UI sections.
    
    Args:
        symptoms: Original patient symptoms
        differential: List of differential diagnoses
        evidence: Retrieved evidence from database
        categories: Categorized evidence by LR utility
        client: OpenAI client instance
        
    Returns:
        Dictionary with 'sections' list containing body system exams
    """
    # Build evidence summary using exam_evidence_free schema field names
    evidence_summary = ""
    
    if categories["high_yield_positive"]:
        evidence_summary += "\nHigh-Yield Positive Findings (LR+ ≥ 5.0):\n"
        for item in categories["high_yield_positive"][:15]:
            evidence_summary += f"- {item.get('original_finding', 'Unknown')} for {item.get('ebm_box_label', 'Unknown')}: LR+ = {item.get('pos_lr_numeric', 'N/A')}\n"
    
    if categories["high_yield_negative"]:
        evidence_summary += "\nHigh-Yield Negative Findings (LR- ≤ 0.2):\n"
        for item in categories["high_yield_negative"][:15]:
            evidence_summary += f"- {item.get('original_finding', 'Unknown')} for {item.get('ebm_box_label', 'Unknown')}: LR- = {item.get('neg_lr_numeric', 'N/A')}\n"
    
    if not evidence_summary:
        evidence_summary = "No specific evidence available in database for these conditions."
    
    prompt = f"""Based on the patient presentation and available evidence, create a focused physical exam strategy organized by body system.

**Patient Presentation:**
{symptoms}

**Differential Diagnosis:**
{', '.join(differential)}

**Available Evidence from McGee's Evidence-Based Physical Diagnosis:**
{evidence_summary}

Return a JSON object with this exact structure:
{{
  "sections": [
    {{
      "system": "Body System Name (e.g., Cardiovascular Exam)",
      "maneuvers": [
        {{
          "name": "Name of the maneuver/finding to assess",
          "purpose": "Brief explanation of what diagnosis this helps rule in or out",
          "lr_info": "LR+ or LR- value if available (e.g., 'LR+ 5.1' or 'LR- 0.1')",
          "technique": "One sentence on how to perform or interpret",
          "high_yield": true or false
        }}
      ]
    }}
  ]
}}

Guidelines:
- Only include body systems relevant to this presentation
- Prioritize high-yield maneuvers (high_yield: true for LR+ ≥ 5 or LR- ≤ 0.2)
- Keep technique descriptions brief and actionable
- Focus on practical bedside assessment
- Do NOT include introduction or conclusion text
- Return ONLY the JSON object, no other text"""

    try:
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert physician educator. Return only valid JSON without any markdown formatting or code blocks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=MAX_TOKENS_SYNTHESIS,
            temperature=TEMPERATURE_SYNTHESIS
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        structured_strategy = parse_strategy_json(content)
        logger.info(f"Generated structured exam strategy with {len(structured_strategy.get('sections', []))} sections")
        return structured_strategy
        
    except Exception as e:
        logger.error(f"Failed to synthesize structured strategy: {e}")
        return {"sections": [], "error": str(e)}


def parse_strategy_json(content: str) -> Dict[str, Any]:
    """
    Parse the JSON strategy response from GPT.
    
    Handles various response formats including markdown code blocks.
    
    Args:
        content: Raw response content from GPT
        
    Returns:
        Parsed dictionary with sections list
    """
    # Remove markdown code blocks if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    content = content.strip()
    
    try:
        result = json.loads(content)
        
        # Validate structure
        if "sections" not in result:
            result = {"sections": []}
        
        # Ensure each section has required fields
        for section in result.get("sections", []):
            if "system" not in section:
                section["system"] = "General Exam"
            if "maneuvers" not in section:
                section["maneuvers"] = []
            
            # Ensure each maneuver has required fields
            for maneuver in section.get("maneuvers", []):
                maneuver.setdefault("name", "Unknown maneuver")
                maneuver.setdefault("purpose", "")
                maneuver.setdefault("lr_info", "")
                maneuver.setdefault("technique", "")
                maneuver.setdefault("high_yield", False)
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse strategy JSON: {e}")
        logger.debug(f"Raw content: {content[:500]}")
        return {"sections": [], "parse_error": str(e)}


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
    # Build evidence summary
    evidence_summary = ""
    
    if categories["high_yield_positive"]:
        evidence_summary += "\n**High-Yield Positive Findings (LR+ ≥ 5.0):**\n"
        for item in categories["high_yield_positive"][:10]:
            evidence_summary += f"- {item.get('original_finding', 'Unknown')} for {item.get('ebm_box_label', 'Unknown')}: LR+ = {item.get('pos_lr_numeric', 'N/A')}\n"
    
    if categories["high_yield_negative"]:
        evidence_summary += "\n**High-Yield Negative Findings (LR- ≤ 0.2):**\n"
        for item in categories["high_yield_negative"][:10]:
            evidence_summary += f"- {item.get('original_finding', 'Unknown')} for {item.get('ebm_box_label', 'Unknown')}: LR- = {item.get('neg_lr_numeric', 'N/A')}\n"
    
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
            max_tokens=MAX_TOKENS_SYNTHESIS,
            temperature=TEMPERATURE_SYNTHESIS
        )
        
        strategy = response.choices[0].message.content.strip()
        logger.info("Generated exam strategy successfully")
        return strategy
        
    except Exception as e:
        logger.error(f"Failed to synthesize strategy: {e}")
        return "Unable to generate strategy. Please try again."


def run_rag_pipeline(symptoms: str) -> Dict[str, Any]:
    """
    Run the complete RAG pipeline for physical exam strategy generation.
    
    Pipeline Steps:
    1. Generate differential diagnosis from symptoms
    2. Retrieve relevant evidence from MongoDB
    3. Categorize evidence by LR utility
    4. Synthesize educational exam strategy (structured by body system)
    
    Args:
        symptoms: Patient symptom description
        
    Returns:
        Dictionary containing:
        - success: bool
        - differential: List of diagnoses
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
        "error": None
    }
    
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
        differential = generate_differential_diagnosis(symptoms, openai_client)
        
        if not differential:
            result["error"] = "Failed to generate differential diagnosis"
            return result
        
        result["differential"] = differential
        
        # Step 2: Retrieve evidence from MongoDB
        logger.info("Step 2: Retrieving evidence from MongoDB...")
        mongo_client = MongoDBClient()
        
        evidence = []
        if mongo_client.connect():
            # Get maneuvers for each diagnosis
            evidence = mongo_client.get_maneuvers_by_diseases(differential)
            
            # Also do text search for symptoms
            symptom_evidence = mongo_client.text_search(symptoms, limit=10)
            
            # Merge and deduplicate using exam_evidence_free schema fields
            seen_findings = set()
            for item in evidence + symptom_evidence:
                finding_key = f"{item.get('ebm_box_label', '')}:{item.get('original_finding', '')}"
                if finding_key not in seen_findings:
                    seen_findings.add(finding_key)
            
            evidence = [e for e in (evidence + symptom_evidence) 
                       if f"{e.get('ebm_box_label', '')}:{e.get('original_finding', '')}" in seen_findings]
            
            mongo_client.close()
        
        logger.info(f"Retrieved {len(evidence)} maneuvers for {len(differential)} diseases")
        result["evidence"] = evidence
        
        if not evidence:
            logger.warning("No evidence retrieved from MongoDB. Database may be empty.")
        
        # Step 3: Categorize evidence
        categories = categorize_evidence(evidence)
        result["categories"] = categories
        
        # Step 4: Synthesize structured strategy (by body system)
        logger.info("Step 4: Synthesizing structured exam strategy...")
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
    
    result["processing_time"] = time.time() - start_time
    logger.info(f"Pipeline completed in {result['processing_time']:.2f} seconds")
    
    return result


def run_rag_pipeline_with_sample_data(symptoms: str) -> Dict[str, Any]:
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
        "using_sample_data": True
    }
    
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
        differential = generate_differential_diagnosis(symptoms, openai_client)
        
        if not differential:
            result["error"] = "Failed to generate differential diagnosis"
            return result
        
        result["differential"] = differential
        
        # Step 2: Filter sample data by differential
        logger.info("Step 2: Filtering sample data by differential...")
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
        categories = categorize_evidence(evidence)
        result["categories"] = categories
        
        # Step 4: Synthesize structured strategy
        logger.info("Step 4: Synthesizing structured exam strategy...")
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
