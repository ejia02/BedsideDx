"""
Chat History Persistence for the McGee AI-PEx Application.

This module implements conversation and message storage for persistent chat
history. It follows the same repository/data-access-layer pattern as
user_auth.py so the storage backend can be swapped from MongoDB to Azure SQL
(via SQLAlchemy) in the future.

Data Model (two collections, SQL-migration-ready):
    conversations — one document per conversation:
        {
            "conversation_id": str (UUID4),
            "username": str (references users.username),
            "title": str (auto-generated, max 80 chars),
            "created_at": datetime (UTC),
            "updated_at": datetime (UTC),
        }

    messages — one document per message:
        {
            "message_id": str (UUID4),
            "conversation_id": str (references conversations.conversation_id),
            "role": str ("user" | "assistant"),
            "message_type": str ("query" | "differential" | "strategy" | "error" | "text"),
            "content_text": str (human-readable content — always populated),
            "content_json": str | null (JSON-serialized structured data),
            "created_at": datetime (UTC),
        }

Security Notes:
    - Every repository method that operates on a conversation verifies that
      the conversation belongs to the authenticated username (authorization
      at the repo level, not just the UI layer).
    - Conversation titles are sanitized to strip HTML tags before storage
      (prevents stored XSS since app.py uses unsafe_allow_html=True).
    - Message content is treated as sensitive — only IDs and counts are logged,
      never full message text.
"""

import re
import json
import uuid
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, OperationFailure
    import certifi
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    logger.warning("pymongo not installed. Install with: pip install pymongo certifi")

# Import configuration
from config import (
    MONGODB_URI,
    DATABASE_NAME,
    CONVERSATIONS_COLLECTION_NAME,
    MESSAGES_COLLECTION_NAME,
    MAX_SIDEBAR_CONVERSATIONS,
    CONVERSATION_TITLE_MAX_LENGTH,
)

# ============================================================================
# Utility Functions
# ============================================================================

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def sanitize_html(text: str) -> str:
    """
    Strip HTML tags from a string to prevent stored XSS.

    Args:
        text: The input string, possibly containing HTML tags.

    Returns:
        The string with all HTML tags removed.
    """
    if not text or not isinstance(text, str):
        return ""
    return _HTML_TAG_PATTERN.sub("", text).strip()


def generate_conversation_title(first_query: str) -> str:
    """
    Generate a conversation title from the first user query.

    The title is sanitized (HTML stripped), truncated to
    CONVERSATION_TITLE_MAX_LENGTH characters, and cleaned up so it does not
    end with a partial word.

    Args:
        first_query: The user's first query in the conversation.

    Returns:
        A cleaned title string.
    """
    if not first_query or not isinstance(first_query, str):
        return "New Conversation"

    title = sanitize_html(first_query).strip()
    # Collapse whitespace
    title = re.sub(r"\s+", " ", title)

    if not title:
        return "New Conversation"

    if len(title) <= CONVERSATION_TITLE_MAX_LENGTH:
        return title

    # Truncate and avoid cutting mid-word
    truncated = title[:CONVERSATION_TITLE_MAX_LENGTH]
    last_space = truncated.rfind(" ")
    if last_space > CONVERSATION_TITLE_MAX_LENGTH // 2:
        truncated = truncated[:last_space]

    return truncated.rstrip(".,;:!?- ") + "..."


# ============================================================================
# Serialization / Deserialization Helpers
# ============================================================================


def serialize_message(session_msg: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    """
    Convert an in-memory session state message dict into the flat database
    format.

    Args:
        session_msg: A message dict from st.session_state.messages.

    Returns:
        Tuple of (message_type, content_text, content_json).
        content_json is None for simple text/error/user messages.
    """
    role = session_msg.get("role", "assistant")
    msg_type = session_msg.get("type", "")

    if role == "user":
        return ("query", session_msg.get("content", ""), None)

    if msg_type == "differential":
        content_text = session_msg.get("content", "")
        items = session_msg.get("items", [])
        content_json = json.dumps(items, default=str) if items else None
        return ("differential", content_text, content_json)

    if msg_type == "strategy":
        result = session_msg.get("result", {})

        # Build a readable text summary
        strategy_text = result.get("strategy", "")
        if not strategy_text:
            # Fallback: generate a brief summary from structured data
            sections = result.get("strategy_structured", {}).get("sections", [])
            parts = ["Physical exam strategy"]
            for sec in sections:
                system_name = sec.get("system", "")
                disease_count = len(sec.get("diseases", []))
                if system_name:
                    parts.append(f"- {system_name}: {disease_count} diseases")
            strategy_text = "\n".join(parts) if len(parts) > 1 else "Physical exam strategy generated."

        # Build the minimal JSON payload — exclude raw evidence & categories
        categories = result.get("categories", {})
        high_yield_count = (
            len(categories.get("high_yield_positive", []))
            + len(categories.get("high_yield_negative", []))
        )
        json_payload = {
            "strategy_structured": result.get("strategy_structured", {}),
            "processing_time": result.get("processing_time", 0),
            "evidence_count": len(result.get("evidence", [])),
            "high_yield_count": high_yield_count,
        }
        content_json = json.dumps(json_payload, default=str)
        return ("strategy", strategy_text, content_json)

    if msg_type == "error":
        return ("error", session_msg.get("content", ""), None)

    # Fallback for plain text or unknown types
    return ("text", session_msg.get("content", ""), None)


def deserialize_message(db_msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruct a session state message dict from a database message document.

    For strategy messages, rebuilds the ``result`` dict with enough structure
    for ``render_strategy_message()`` to work. Falls back to rendering
    content_text as plain markdown if the JSON is corrupt.

    Args:
        db_msg: A message document from the messages collection.

    Returns:
        A dict compatible with st.session_state.messages.
    """
    role = db_msg.get("role", "assistant")
    message_type = db_msg.get("message_type", "text")
    content_text = db_msg.get("content_text", "")
    content_json = db_msg.get("content_json")

    if role == "user" or message_type == "query":
        return {"role": "user", "content": content_text}

    if message_type == "differential":
        items = []
        if content_json:
            try:
                items = json.loads(content_json)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to parse differential content_json for message %s",
                    db_msg.get("message_id", "unknown"),
                )
        return {
            "role": "assistant",
            "type": "differential",
            "content": content_text,
            "items": items,
        }

    if message_type == "strategy":
        # Try to reconstruct the result dict from content_json
        if content_json:
            try:
                payload = json.loads(content_json)
                result = {
                    "success": True,
                    "strategy_structured": payload.get("strategy_structured", {}),
                    "processing_time": payload.get("processing_time", 0),
                    "strategy": content_text,
                    "evidence": [],  # Stub — raw evidence is not persisted
                    "categories": {
                        "high_yield_positive": [{}] * payload.get("high_yield_count", 0)
                        if payload.get("high_yield_count")
                        else [],
                        "high_yield_negative": [],
                    },
                }
                return {
                    "role": "assistant",
                    "type": "strategy",
                    "result": result,
                }
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to parse strategy content_json for message %s, "
                    "falling back to plain text",
                    db_msg.get("message_id", "unknown"),
                )

        # Fallback: render as plain markdown
        return {
            "role": "assistant",
            "type": "differential",  # render as markdown
            "content": content_text,
            "items": [],
        }

    if message_type == "error":
        return {
            "role": "assistant",
            "type": "error",
            "content": content_text,
        }

    # Fallback for "text" or unknown
    return {
        "role": "assistant",
        "content": content_text,
    }


# ============================================================================
# Chat History Repository — Abstract Base Class
# ============================================================================


class ChatHistoryRepository(ABC):
    """
    Abstract base class defining the interface for chat history data access.

    Implement this interface to swap storage backends (e.g., MongoDB -> Azure SQL).
    All methods use flat dictionaries with simple CRUD semantics so they map
    cleanly to relational tables.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish a connection to the data store. Returns True on success."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the connection to the data store."""
        ...

    @abstractmethod
    def create_conversation(self, username: str, title: str) -> Optional[Dict[str, Any]]:
        """
        Create a new conversation.

        Args:
            username: The owner's username.
            title: The conversation title.

        Returns:
            The conversation dict with conversation_id, or None on failure.
        """
        ...

    @abstractmethod
    def get_conversations_for_user(
        self, username: str, limit: int = MAX_SIDEBAR_CONVERSATIONS
    ) -> List[Dict[str, Any]]:
        """
        List conversations for a user, sorted by updated_at descending.

        Args:
            username: The owner's username.
            limit: Maximum number of conversations to return.

        Returns:
            List of conversation dicts, most recent first.
        """
        ...

    @abstractmethod
    def get_conversation(
        self, conversation_id: str, username: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single conversation's metadata.

        Args:
            conversation_id: The conversation to retrieve.
            username: The requesting user (for authorization).

        Returns:
            Conversation dict or None if not found / unauthorized.
        """
        ...

    @abstractmethod
    def get_messages(
        self, conversation_id: str, username: str
    ) -> List[Dict[str, Any]]:
        """
        Get all messages for a conversation, sorted chronologically.

        Args:
            conversation_id: The conversation to retrieve messages for.
            username: The requesting user (for authorization).

        Returns:
            List of message dicts sorted by created_at ascending.
        """
        ...

    @abstractmethod
    def add_message(
        self,
        conversation_id: str,
        username: str,
        role: str,
        message_type: str,
        content_text: str,
        content_json: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Insert a message and update the conversation's updated_at timestamp.

        Args:
            conversation_id: The conversation to add the message to.
            username: The requesting user (for authorization).
            role: "user" or "assistant".
            message_type: "query", "differential", "strategy", "error", or "text".
            content_text: Human-readable text content.
            content_json: Optional JSON-serialized structured data.

        Returns:
            The message dict with message_id, or None on failure.
        """
        ...

    @abstractmethod
    def update_conversation_title(
        self, conversation_id: str, username: str, title: str
    ) -> bool:
        """
        Update a conversation's title.

        Args:
            conversation_id: The conversation to update.
            username: The requesting user (for authorization).
            title: The new title.

        Returns:
            True if updated, False otherwise.
        """
        ...

    @abstractmethod
    def delete_conversation(self, conversation_id: str, username: str) -> bool:
        """
        Delete a conversation and all its messages (cascade delete).

        Args:
            conversation_id: The conversation to delete.
            username: The requesting user (for authorization).

        Returns:
            True if deleted, False otherwise.
        """
        ...

    @abstractmethod
    def delete_all_conversations_for_user(self, username: str) -> int:
        """
        Delete all conversations and their messages for a user.

        Args:
            username: The user whose conversations to delete.

        Returns:
            Count of deleted conversations.
        """
        ...


# ============================================================================
# MongoDB Chat History Repository
# ============================================================================


class MongoChatHistoryRepository(ChatHistoryRepository):
    """
    MongoDB-backed implementation of ChatHistoryRepository.

    Stores conversations and messages in two separate collections within the
    ``bedside_dx`` database. The document schemas use flat, normalized fields
    so they map cleanly to relational tables for future Azure SQL migration.

    Authorization:
        Every method that accepts a conversation_id also requires a username
        and verifies ownership before proceeding.
    """

    def __init__(
        self,
        uri: str = None,
        database_name: str = None,
        conversations_collection: str = None,
        messages_collection: str = None,
    ):
        """
        Initialize the MongoDB chat history repository.

        Args:
            uri: MongoDB connection string (defaults to config MONGODB_URI).
            database_name: Database name (defaults to config DATABASE_NAME).
            conversations_collection: Collection name for conversations.
            messages_collection: Collection name for messages.
        """
        self.uri = uri or MONGODB_URI
        self.database_name = database_name or DATABASE_NAME
        self.conversations_collection_name = (
            conversations_collection or CONVERSATIONS_COLLECTION_NAME
        )
        self.messages_collection_name = (
            messages_collection or MESSAGES_COLLECTION_NAME
        )
        self.client = None
        self.db = None
        self.conversations = None
        self.messages = None
        self._connected = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Establish connection to MongoDB and ensure indexes exist.

        Returns:
            True if connection successful, False otherwise.
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
                tlsCAFile=certifi.where(),
            )
            # Test connection
            self.client.admin.command("ping")

            self.db = self.client[self.database_name]
            self.conversations = self.db[self.conversations_collection_name]
            self.messages = self.db[self.messages_collection_name]
            self._connected = True

            logger.info(
                "Connected to MongoDB chat history: %s.{%s, %s}",
                self.database_name,
                self.conversations_collection_name,
                self.messages_collection_name,
            )

            self._ensure_indexes()
            return True

        except ConnectionFailure as e:
            logger.error("Failed to connect to MongoDB for chat history: %s", e)
            return False
        except Exception as e:
            logger.error("MongoDB chat history connection error: %s", e)
            return False

    def _ensure_indexes(self) -> None:
        """Ensure required indexes exist on conversations and messages collections."""
        try:
            # --- Conversations indexes ---
            conv_indexes = self.conversations.index_information()

            if "conversation_id_unique" not in conv_indexes:
                self.conversations.create_index(
                    [("conversation_id", ASCENDING)],
                    unique=True,
                    name="conversation_id_unique",
                )
                logger.info("Created unique index on conversations.conversation_id")

            if "username_idx" not in conv_indexes:
                self.conversations.create_index(
                    [("username", ASCENDING)],
                    name="username_idx",
                )
                logger.info("Created index on conversations.username")

            if "updated_at_idx" not in conv_indexes:
                self.conversations.create_index(
                    [("updated_at", DESCENDING)],
                    name="updated_at_idx",
                )
                logger.info("Created index on conversations.updated_at")

            # --- Messages indexes ---
            msg_indexes = self.messages.index_information()

            if "message_id_unique" not in msg_indexes:
                self.messages.create_index(
                    [("message_id", ASCENDING)],
                    unique=True,
                    name="message_id_unique",
                )
                logger.info("Created unique index on messages.message_id")

            if "conversation_id_idx" not in msg_indexes:
                self.messages.create_index(
                    [("conversation_id", ASCENDING)],
                    name="conversation_id_idx",
                )
                logger.info("Created index on messages.conversation_id")

            if "conv_created_compound" not in msg_indexes:
                self.messages.create_index(
                    [("conversation_id", ASCENDING), ("created_at", ASCENDING)],
                    name="conv_created_compound",
                )
                logger.info("Created compound index on messages.(conversation_id, created_at)")

        except OperationFailure as e:
            logger.warning("Could not create chat history indexes (may already exist): %s", e)
        except Exception as e:
            logger.warning("Chat history index creation warning: %s", e)

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("MongoDB chat history connection closed")

    def is_connected(self) -> bool:
        """Check if the repository is connected."""
        return self._connected and self.conversations is not None and self.messages is not None

    # ------------------------------------------------------------------
    # Authorization helper
    # ------------------------------------------------------------------

    def _verify_conversation_owner(
        self, conversation_id: str, username: str
    ) -> Optional[Dict[str, Any]]:
        """
        Verify that the given conversation belongs to the given user.

        Args:
            conversation_id: The conversation ID to check.
            username: The expected owner.

        Returns:
            The conversation document if authorized, None otherwise.
        """
        try:
            conv = self.conversations.find_one(
                {"conversation_id": conversation_id, "username": username},
                {"_id": 0},
            )
            if conv is None:
                logger.warning(
                    "Authorization failed: conversation %s not found for user %s",
                    conversation_id,
                    username,
                )
            return conv
        except Exception as e:
            logger.error("Authorization check failed: %s", type(e).__name__)
            return None

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def create_conversation(
        self, username: str, title: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new conversation.

        Args:
            username: The owner's username.
            title: The conversation title (will be sanitized).

        Returns:
            The conversation dict with conversation_id, or None on failure.
        """
        if not self.is_connected():
            return None

        now = datetime.now(timezone.utc)
        conversation_id = str(uuid.uuid4())
        safe_title = sanitize_html(title)[:CONVERSATION_TITLE_MAX_LENGTH]

        conv_doc = {
            "conversation_id": conversation_id,
            "username": username,
            "title": safe_title,
            "created_at": now,
            "updated_at": now,
        }

        try:
            self.conversations.insert_one(conv_doc)
            logger.info("Conversation created: %s for user %s", conversation_id, username)
            return {k: v for k, v in conv_doc.items() if k != "_id"}
        except Exception as e:
            logger.error(
                "Failed to create conversation for user %s: %s",
                username,
                type(e).__name__,
            )
            return None

    def get_conversations_for_user(
        self, username: str, limit: int = MAX_SIDEBAR_CONVERSATIONS
    ) -> List[Dict[str, Any]]:
        """
        List conversations for a user, sorted by updated_at descending.

        Args:
            username: The owner's username.
            limit: Maximum number of conversations to return.

        Returns:
            List of conversation dicts, most recent first.
        """
        if not self.is_connected():
            return []

        try:
            cursor = (
                self.conversations.find(
                    {"username": username},
                    {"_id": 0},
                )
                .sort("updated_at", DESCENDING)
                .limit(limit)
            )
            results = list(cursor)
            logger.info(
                "Retrieved %d conversations for user %s", len(results), username
            )
            return results
        except Exception as e:
            logger.error(
                "Failed to get conversations for user %s: %s",
                username,
                type(e).__name__,
            )
            return []

    def get_conversation(
        self, conversation_id: str, username: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single conversation's metadata.

        Args:
            conversation_id: The conversation to retrieve.
            username: The requesting user (for authorization).

        Returns:
            Conversation dict or None if not found / unauthorized.
        """
        if not self.is_connected():
            return None

        return self._verify_conversation_owner(conversation_id, username)

    def get_messages(
        self, conversation_id: str, username: str
    ) -> List[Dict[str, Any]]:
        """
        Get all messages for a conversation, sorted chronologically.

        Args:
            conversation_id: The conversation to retrieve messages for.
            username: The requesting user (for authorization).

        Returns:
            List of message dicts sorted by created_at ascending.
        """
        if not self.is_connected():
            return []

        # Authorization check
        if self._verify_conversation_owner(conversation_id, username) is None:
            return []

        try:
            cursor = (
                self.messages.find(
                    {"conversation_id": conversation_id},
                    {"_id": 0},
                )
                .sort("created_at", ASCENDING)
            )
            results = list(cursor)
            logger.info(
                "Retrieved %d messages for conversation %s",
                len(results),
                conversation_id,
            )
            return results
        except Exception as e:
            logger.error(
                "Failed to get messages for conversation %s: %s",
                conversation_id,
                type(e).__name__,
            )
            return []

    def add_message(
        self,
        conversation_id: str,
        username: str,
        role: str,
        message_type: str,
        content_text: str,
        content_json: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Insert a message and update the conversation's updated_at timestamp.

        Args:
            conversation_id: The conversation to add the message to.
            username: The requesting user (for authorization).
            role: "user" or "assistant".
            message_type: "query", "differential", "strategy", "error", or "text".
            content_text: Human-readable text content.
            content_json: Optional JSON-serialized structured data.

        Returns:
            The message dict with message_id, or None on failure.
        """
        if not self.is_connected():
            return None

        # Authorization check
        if self._verify_conversation_owner(conversation_id, username) is None:
            return None

        now = datetime.now(timezone.utc)
        message_id = str(uuid.uuid4())

        msg_doc = {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "message_type": message_type,
            "content_text": content_text,
            "content_json": content_json,
            "created_at": now,
        }

        try:
            self.messages.insert_one(msg_doc)

            # Update conversation's updated_at
            self.conversations.update_one(
                {"conversation_id": conversation_id},
                {"$set": {"updated_at": now}},
            )

            logger.info(
                "Message %s added to conversation %s (type=%s)",
                message_id,
                conversation_id,
                message_type,
            )
            return {k: v for k, v in msg_doc.items() if k != "_id"}
        except Exception as e:
            logger.error(
                "Failed to add message to conversation %s: %s",
                conversation_id,
                type(e).__name__,
            )
            return None

    def update_conversation_title(
        self, conversation_id: str, username: str, title: str
    ) -> bool:
        """
        Update a conversation's title.

        Args:
            conversation_id: The conversation to update.
            username: The requesting user (for authorization).
            title: The new title (will be sanitized).

        Returns:
            True if updated, False otherwise.
        """
        if not self.is_connected():
            return False

        # Authorization check
        if self._verify_conversation_owner(conversation_id, username) is None:
            return False

        safe_title = sanitize_html(title)[:CONVERSATION_TITLE_MAX_LENGTH]

        try:
            result = self.conversations.update_one(
                {"conversation_id": conversation_id, "username": username},
                {"$set": {"title": safe_title, "updated_at": datetime.now(timezone.utc)}},
            )
            if result.matched_count > 0:
                logger.info(
                    "Conversation %s title updated for user %s",
                    conversation_id,
                    username,
                )
                return True
            return False
        except Exception as e:
            logger.error(
                "Failed to update conversation %s title: %s",
                conversation_id,
                type(e).__name__,
            )
            return False

    def delete_conversation(self, conversation_id: str, username: str) -> bool:
        """
        Delete a conversation and all its messages (cascade delete).

        Args:
            conversation_id: The conversation to delete.
            username: The requesting user (for authorization).

        Returns:
            True if deleted, False otherwise.
        """
        if not self.is_connected():
            return False

        # Authorization check
        if self._verify_conversation_owner(conversation_id, username) is None:
            return False

        try:
            # Delete messages first (cascade)
            msg_result = self.messages.delete_many(
                {"conversation_id": conversation_id}
            )
            logger.info(
                "Deleted %d messages for conversation %s",
                msg_result.deleted_count,
                conversation_id,
            )

            # Delete the conversation
            conv_result = self.conversations.delete_one(
                {"conversation_id": conversation_id, "username": username}
            )
            if conv_result.deleted_count > 0:
                logger.info(
                    "Conversation %s deleted for user %s",
                    conversation_id,
                    username,
                )
                return True
            return False
        except Exception as e:
            logger.error(
                "Failed to delete conversation %s: %s",
                conversation_id,
                type(e).__name__,
            )
            return False

    def delete_all_conversations_for_user(self, username: str) -> int:
        """
        Delete all conversations and their messages for a user.

        Args:
            username: The user whose conversations to delete.

        Returns:
            Count of deleted conversations.
        """
        if not self.is_connected():
            return 0

        try:
            # Get all conversation IDs for the user
            conv_ids = [
                doc["conversation_id"]
                for doc in self.conversations.find(
                    {"username": username},
                    {"conversation_id": 1, "_id": 0},
                )
            ]

            if not conv_ids:
                return 0

            # Delete all messages for those conversations
            msg_result = self.messages.delete_many(
                {"conversation_id": {"$in": conv_ids}}
            )
            logger.info(
                "Deleted %d messages across %d conversations for user %s",
                msg_result.deleted_count,
                len(conv_ids),
                username,
            )

            # Delete all conversations
            conv_result = self.conversations.delete_many(
                {"username": username}
            )
            logger.info(
                "Deleted %d conversations for user %s",
                conv_result.deleted_count,
                username,
            )
            return conv_result.deleted_count
        except Exception as e:
            logger.error(
                "Failed to delete all conversations for user %s: %s",
                username,
                type(e).__name__,
            )
            return 0
