"""
Free Vector Store for Physical Exam Evidence
Uses local Sentence Transformers instead of OpenAI (no API costs!)
"""

import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import asdict
import numpy as np

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from excel_parser import ExamDocument, ExcelParser

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FreeVectorStore:
    """MongoDB-based vector store using free local embeddings"""
    
    def __init__(self, 
                 mongodb_uri: str = None,
                 database_name: str = "bedside_dx",
                 collection_name: str = "mcgee_evidence",
                 model_name: str = "all-MiniLM-L6-v2"):
        
        # Set up local embedding model (free!)
        self.model_name = model_name
        logger.info(f"Loading local embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
        # Set up MongoDB
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI")
        if not self.mongodb_uri:
            raise ValueError("MongoDB URI required. Set MONGODB_URI environment variable.")
        
        self.database_name = database_name
        self.collection_name = collection_name
        
        # Initialize connections
        self.client: MongoClient = None
        self.database: Database = None
        self.collection: Collection = None
        
        self._connect()
    
    def _connect(self) -> None:
        """Connect to MongoDB"""
        try:
            # Try standard connection first
            self.client = MongoClient(
                self.mongodb_uri,
                serverSelectionTimeoutMS=20000,
                connectTimeoutMS=20000,
                socketTimeoutMS=20000
            )
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self.database_name}.{self.collection_name}")
            
        except Exception as e:
            logger.warning(f"Standard connection failed: {e}")
            # Try with relaxed SSL for macOS compatibility (development only)
            try:
                logger.info("Attempting connection with relaxed SSL settings (macOS compatibility)...")
                self.client = MongoClient(
                    self.mongodb_uri,
                    tlsAllowInvalidCertificates=True,  # Workaround for macOS SSL issues
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=30000,
                    socketTimeoutMS=30000
                )
                self.database = self.client[self.database_name]
                self.collection = self.database[self.collection_name]
                self.client.admin.command('ping')
                logger.info(f"Connected to MongoDB (relaxed SSL): {self.database_name}.{self.collection_name}")
            except Exception as e2:
                logger.error(f"All connection attempts failed: {e2}")
                logger.error("\nTROUBLESHOOTING:")
                logger.error("1. Check MongoDB Atlas cluster is running")
                logger.error("2. Verify Network Access - whitelist your IP (0.0.0.0/0 for testing)")
                logger.error("3. Check your password in the connection string")
                logger.error("4. Try getting a fresh connection string from MongoDB Atlas")
                raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using local Sentence Transformer (FREE!)"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches (FREE!)"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Generating embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                # Process batch locally (no API calls!)
                batch_embeddings = self.embedding_model.encode(batch)
                embeddings.extend([emb.tolist() for emb in batch_embeddings])
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                raise
        
        return embeddings
    
    def create_vector_search_index_info(self) -> None:
        """Show vector search index configuration for the free model"""
        import json
        
        # Main embedding index (for full text search)
        main_index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": self.embedding_dim,  # 384 for all-MiniLM-L6-v2
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
        
        # Disease-focused embedding index (for ebm_box_label vector search)
        label_index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "ebm_box_label_embedding",
                    "numDimensions": self.embedding_dim,  # 384 for all-MiniLM-L6-v2
                    "similarity": "cosine"
                }
            ]
        }
        
        logger.info("Vector search index definitions for FREE model:")
        logger.info(f"Embedding dimensions: {self.embedding_dim}")
        
        print("\n" + "="*60)
        print("INDEX 1: Main Vector Search Index (vector_index_free)")
        print("For searching by full text embedding (maneuvers, findings, etc.)")
        print("="*60)
        print(json.dumps(main_index_definition, indent=2))
        
        print("\n" + "="*60)
        print("INDEX 2: Disease Label Vector Search Index (ebm_label_vector_index)")
        print("For disease-focused semantic search on ebm_box_label")
        print("="*60)
        print(json.dumps(label_index_definition, indent=2))
    
    def _transform_to_nested_schema(self, doc: ExamDocument) -> Dict[str, Any]:
        """Transform flat ExamDocument to nested schema for rag_engine.py compatibility"""
        return {
            "source": {
                "ebm_box_label": doc.ebm_box_label,
                "chapter": doc.chapter,
                "ebm_box_id": doc.ebm_box_id,
            },
            "maneuver": {
                "name": doc.maneuver_base,
                "normalized": doc.maneuver_base.lower() if doc.maneuver_base else None,
            },
            "claim": {
                "summary": None,
                "target_condition": doc.ebm_box_label,
            },
            "context": {
                "conditions": [],
            },
            "result_buckets": [
                {
                    "label": doc.result_modifier or "present",
                    "lr_positive": doc.pos_lr_numeric,
                    "lr_negative": doc.neg_lr_numeric,
                    "pretest_prob": doc.pretest_prob_numeric,
                }
            ],
            "text_for_embedding": doc.text_for_embedding,
            "original_finding": doc.original_finding,
        }
    
    def store_documents(self, documents: List[ExamDocument]) -> None:
        """Store documents with embeddings in MongoDB (FREE!)"""
        if not documents:
            logger.warning("No documents to store")
            return
        
        # Generate embeddings for text_for_embedding (full context)
        texts = [doc.text_for_embedding for doc in documents]
        logger.info(f"Generating FREE local embeddings for {len(texts)} documents...")
        embeddings = self.generate_embeddings_batch(texts)
        
        # Generate embeddings for ebm_box_label (disease names) for disease-focused search
        ebm_labels = [doc.ebm_box_label for doc in documents]
        logger.info(f"Generating ebm_box_label embeddings for {len(ebm_labels)} disease labels...")
        label_embeddings = self.generate_embeddings_batch(ebm_labels)
        
        # Prepare documents for insertion using nested schema
        mongo_docs = []
        for doc, embedding, label_embedding in zip(documents, embeddings, label_embeddings):
            # Transform to nested schema for rag_engine.py compatibility
            mongo_doc = self._transform_to_nested_schema(doc)
            mongo_doc['embedding'] = embedding
            mongo_doc['ebm_box_label_embedding'] = label_embedding  # Disease-focused embedding
            mongo_docs.append(mongo_doc)
        
        # Insert documents
        try:
            result = self.collection.insert_many(mongo_docs)
            logger.info(f"Inserted {len(result.inserted_ids)} documents into MongoDB")
            
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        try:
            result = self.collection.delete_many({})
            logger.info(f"Deleted {result.deleted_count} documents from collection")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def vector_search(self, 
                     query: str, 
                     limit: int = 10,
                     ebm_box_filter: Optional[str] = None,
                     chapter_filter: Optional[str] = None,
                     maneuver_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform vector search with optional filtering (FREE!)
        """
        # Generate query embedding locally (no API cost!)
        query_embedding = self.generate_embedding(query)
        
        # Build aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index_free",  # Different index name for free model
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            }
        ]
        
        # Add filters if provided (using nested schema paths)
        match_conditions = {}
        if ebm_box_filter:
            match_conditions["source.ebm_box_label"] = {"$regex": ebm_box_filter, "$options": "i"}
        if chapter_filter:
            match_conditions["source.chapter"] = {"$regex": chapter_filter, "$options": "i"}
        if maneuver_filter:
            match_conditions["maneuver.name"] = {"$regex": maneuver_filter, "$options": "i"}
        
        if match_conditions:
            pipeline.append({"$match": match_conditions})
        
        # Add score and project fields
        pipeline.extend([
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
        ])
        
        try:
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Vector search returned {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            # Fallback to text search if vector search fails
            return self._fallback_text_search(query, limit, match_conditions)
    
    def disease_vector_search(self, 
                              query: str, 
                              limit: int = 20) -> List[Dict[str, Any]]:
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
        # Generate query embedding locally (no API cost!)
        query_embedding = self.generate_embedding(query)
        
        # Build aggregation pipeline for disease-focused search
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
        
        try:
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Disease vector search returned {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Disease vector search failed: {e}")
            # Fallback to regex search on ebm_box_label
            return self._fallback_disease_search(query, limit)
    
    def _fallback_disease_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback regex search for disease names when vector search is unavailable"""
        logger.info("Using fallback disease search (regex)")
        
        try:
            results = list(self.collection.find(
                {"source.ebm_box_label": {"$regex": query, "$options": "i"}},
                {"embedding": 0, "ebm_box_label_embedding": 0}
            ).limit(limit))
            
            # Add dummy score
            for result in results:
                result["score"] = 0.5
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback disease search failed: {e}")
            return []
    
    def _fallback_text_search(self, 
                             query: str, 
                             limit: int,
                             match_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback text search when vector search is not available"""
        logger.info("Using fallback text search")
        
        # Build text search query (using nested schema paths)
        search_conditions = {
            "$or": [
                {"text_for_embedding": {"$regex": query, "$options": "i"}},
                {"original_finding": {"$regex": query, "$options": "i"}},
                {"maneuver.name": {"$regex": query, "$options": "i"}},
                {"source.ebm_box_label": {"$regex": query, "$options": "i"}}
            ]
        }
        
        # Combine with filters
        if match_conditions:
            search_conditions = {"$and": [search_conditions, match_conditions]}
        
        try:
            results = list(self.collection.find(
                search_conditions,
                {"embedding": 0, "ebm_box_label_embedding": 0}  # Exclude embeddings
            ).limit(limit))
            
            # Add dummy score
            for result in results:
                result["score"] = 0.5
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback text search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            stats = {
                "total_documents": self.collection.count_documents({}),
                "unique_chapters": len(self.collection.distinct("source.chapter")),
                "unique_ebm_boxes": len(self.collection.distinct("source.ebm_box_label")),
                "unique_maneuvers": len(self.collection.distinct("maneuver.name")),
                "embedding_model": self.model_name,
                "embedding_dimensions": self.embedding_dim
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}


def main():
    """Test the free vector store"""
    # Load documents
    parser = ExcelParser('/Users/ericjia/Downloads/BedsideDx/AppendixChp71_Table71_1 (1).xlsx')
    documents = parser.parse_documents()
    
    print(f"Loaded {len(documents)} documents")
    
    # Note: You only need MongoDB URI (no OpenAI key needed!)
    
    try:
        # Initialize FREE vector store
        vector_store = FreeVectorStore()
        
        # Show index configuration
        vector_store.create_vector_search_index_info()
        
        # Test with first 5 documents
        print(f"\nTesting with first 5 documents...")
        vector_store.clear_collection()
        vector_store.store_documents(documents[:5])
        
        # Get stats
        stats = vector_store.get_collection_stats()
        print("Collection stats:", stats)
        
        # Test general vector search
        results = vector_store.vector_search("heart murmur aortic stenosis", limit=3)
        print(f"\nGeneral vector search for 'heart murmur aortic stenosis':")
        for i, result in enumerate(results):
            source = result.get('source', {})
            buckets = result.get('result_buckets', [{}])
            bucket = buckets[0] if buckets else {}
            print(f"{i+1}. {result.get('original_finding', 'N/A')} (Score: {result.get('score', 0):.3f})")
            print(f"   EBM Box: {source.get('ebm_box_label', 'N/A')}")
            print(f"   LR+: {bucket.get('lr_positive', 'N/A')}, LR-: {bucket.get('lr_negative', 'N/A')}")
        
        # Test disease-focused vector search
        print(f"\n" + "="*50)
        print("Testing disease-focused vector search...")
        disease_results = vector_store.disease_vector_search("visual field defects", limit=3)
        print(f"\nDisease vector search for 'visual field defects':")
        for i, result in enumerate(disease_results):
            source = result.get('source', {})
            buckets = result.get('result_buckets', [{}])
            bucket = buckets[0] if buckets else {}
            print(f"{i+1}. {result.get('original_finding', 'N/A')} (Score: {result.get('score', 0):.3f})")
            print(f"   EBM Box: {source.get('ebm_box_label', 'N/A')}")
            print(f"   LR+: {bucket.get('lr_positive', 'N/A')}, LR-: {bucket.get('lr_negative', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set up your MongoDB URI in .env file")


if __name__ == "__main__":
    main()

