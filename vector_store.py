"""
Vector Store for Physical Exam Evidence
Handles embedding generation and MongoDB storage with vector search
"""

import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import asdict
import asyncio

import openai
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv

from excel_parser import ExamDocument, ExcelParser

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """MongoDB-based vector store for physical exam evidence"""
    
    def __init__(self, 
                 mongodb_uri: str = None,
                 database_name: str = "bedside_dx",
                 collection_name: str = "exam_evidence",
                 openai_api_key: str = None):
        
        # Set up OpenAI
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.openai_api_key
        
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
            self.client = MongoClient(self.mongodb_uri)
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self.database_name}.{self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI text-embedding-3-small"""
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Generating embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                response = openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                raise
        
        return embeddings
    
    def create_vector_search_index(self) -> None:
        """Create vector search index in MongoDB Atlas"""
        index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1536,  # text-embedding-3-small dimension
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
        
        try:
            # Note: This requires MongoDB Atlas with vector search enabled
            # The actual index creation needs to be done through Atlas UI or Atlas CLI
            logger.info("Vector search index definition:")
            logger.info(index_definition)
            logger.info("Please create this index in MongoDB Atlas manually or via Atlas CLI")
            
        except Exception as e:
            logger.error(f"Note: Vector search index creation failed: {e}")
            logger.info("You may need to create the index manually in MongoDB Atlas")
    
    def store_documents(self, documents: List[ExamDocument]) -> None:
        """Store documents with embeddings in MongoDB"""
        if not documents:
            logger.warning("No documents to store")
            return
        
        # Generate embeddings
        texts = [doc.text_for_embedding for doc in documents]
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.generate_embeddings_batch(texts)
        
        # Prepare documents for insertion
        mongo_docs = []
        for doc, embedding in zip(documents, embeddings):
            mongo_doc = asdict(doc)
            mongo_doc['embedding'] = embedding
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
        Perform vector search with optional filtering
        
        Args:
            query: Search query text
            limit: Number of results to return
            ebm_box_filter: Filter by EBM box label
            chapter_filter: Filter by chapter
            maneuver_filter: Filter by maneuver base
        
        Returns:
            List of matching documents with scores
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Build aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",  # Name of your vector search index
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,  # Search more candidates for better results
                    "limit": limit
                }
            }
        ]
        
        # Add filters if provided
        match_conditions = {}
        if ebm_box_filter:
            match_conditions["ebm_box_label"] = {"$regex": ebm_box_filter, "$options": "i"}
        if chapter_filter:
            match_conditions["chapter"] = {"$regex": chapter_filter, "$options": "i"}
        if maneuver_filter:
            match_conditions["maneuver_base"] = {"$regex": maneuver_filter, "$options": "i"}
        
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
    
    def _fallback_text_search(self, 
                             query: str, 
                             limit: int,
                             match_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback text search when vector search is not available"""
        logger.info("Using fallback text search")
        
        # Build text search query
        search_conditions = {
            "$or": [
                {"text_for_embedding": {"$regex": query, "$options": "i"}},
                {"original_finding": {"$regex": query, "$options": "i"}},
                {"maneuver_base": {"$regex": query, "$options": "i"}},
                {"ebm_box_label": {"$regex": query, "$options": "i"}}
            ]
        }
        
        # Combine with filters
        if match_conditions:
            search_conditions = {"$and": [search_conditions, match_conditions]}
        
        try:
            results = list(self.collection.find(
                search_conditions,
                {"embedding": 0}  # Exclude embedding
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
                "unique_chapters": len(self.collection.distinct("chapter")),
                "unique_ebm_boxes": len(self.collection.distinct("ebm_box_label")),
                "unique_maneuvers": len(self.collection.distinct("maneuver_base"))
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}


def main():
    """Test the vector store"""
    # Load documents
    parser = ExcelParser('/Users/ericjia/Downloads/BedsideDx/AppendixChp71_Table71_1 (1).xlsx')
    documents = parser.parse_documents()
    
    print(f"Loaded {len(documents)} documents")
    
    # Note: You need to set up environment variables first
    # Create .env file with:
    # OPENAI_API_KEY=your_openai_key
    # MONGODB_URI=your_mongodb_atlas_connection_string
    
    try:
        # Initialize vector store
        vector_store = VectorStore()
        
        # Clear existing data (optional)
        # vector_store.clear_collection()
        
        # Store documents (this will take a while due to embedding generation)
        # vector_store.store_documents(documents[:10])  # Test with first 10 documents
        
        # Get stats
        stats = vector_store.get_collection_stats()
        print("Collection stats:", stats)
        
        # Test search
        results = vector_store.vector_search("heart murmur aortic stenosis", limit=5)
        print(f"\nSearch results for 'heart murmur aortic stenosis':")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.get('original_finding', 'N/A')} (Score: {result.get('score', 0):.3f})")
            print(f"   EBM Box: {result.get('ebm_box_label', 'N/A')}")
            print(f"   LR+: {result.get('pos_lr_numeric', 'N/A')}, LR-: {result.get('neg_lr_numeric', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set up your .env file with OPENAI_API_KEY and MONGODB_URI")


if __name__ == "__main__":
    main()

