"""
User Authentication and Account Management for the McGee AI-PEx Application.

This module implements user registration, login, profile management,
password hashing, input validation, and rate limiting. It is designed
with a repository/data-access-layer pattern so the storage backend
can be swapped from MongoDB to Azure SQL (via SQLAlchemy) in the future.

Security Notes:
    - Passwords are hashed with bcrypt before storage; plaintext is never persisted.
    - Phone numbers and emails are PII. They must NEVER be logged, included in
      error messages, or exposed in the UI beyond the user's own profile view.
    - In production, enable encryption at rest (Azure SQL TDE or MongoDB
      Encrypted Storage Engine) to protect PII fields.

Future query_history collection/table schema (reference only — not implemented):
    {
        "query_id": str (UUID),
        "username": str (foreign key to users.username),
        "query_text": str,
        "result_summary": str,
        "notification_sent": bool,
        "created_at": datetime
    }
"""

import re
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
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not installed. Install with: pip install bcrypt")

try:
    from pymongo import MongoClient, ASCENDING
    from pymongo.errors import ConnectionFailure, OperationFailure, DuplicateKeyError
    import certifi
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    logger.warning("pymongo not installed. Install with: pip install pymongo certifi")

# Import configuration
from config import (
    MONGODB_URI,
    DATABASE_NAME,
    USERS_COLLECTION_NAME,
    LOGIN_ATTEMPTS_COLLECTION_NAME,
    MAX_LOGIN_ATTEMPTS,
    LOGIN_ATTEMPT_WINDOW_MINUTES,
    PASSWORD_MIN_LENGTH,
    USERNAME_MIN_LENGTH,
    USERNAME_MAX_LENGTH,
)

# ============================================================================
# Input Validation
# ============================================================================

# Regex patterns
_EMAIL_PATTERN = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)
_PHONE_PATTERN = re.compile(
    r"^\+[1-9]\d{6,14}$"  # E.164: + followed by 7-15 digits, no leading zero
)
_USERNAME_PATTERN = re.compile(
    r"^[a-zA-Z0-9_]+$"  # Alphanumeric + underscores only
)


def validate_username(username: str) -> Tuple[bool, str]:
    """
    Validate a username string.

    Rules:
        - Between USERNAME_MIN_LENGTH and USERNAME_MAX_LENGTH characters
        - Alphanumeric characters and underscores only
        - Stripped of leading/trailing whitespace

    Args:
        username: The username to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty on success.
    """
    if not username or not isinstance(username, str):
        return False, "Username is required."

    username = username.strip()

    if len(username) < USERNAME_MIN_LENGTH:
        return False, f"Username must be at least {USERNAME_MIN_LENGTH} characters."
    if len(username) > USERNAME_MAX_LENGTH:
        return False, f"Username must be at most {USERNAME_MAX_LENGTH} characters."
    if not _USERNAME_PATTERN.match(username):
        return False, "Username may only contain letters, numbers, and underscores."

    return True, ""


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate an email address format.

    Args:
        email: The email address to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not email or not isinstance(email, str):
        return False, "Email address is required."

    email = email.strip().lower()

    if not _EMAIL_PATTERN.match(email):
        return False, "Invalid email address format."

    return True, ""


def validate_phone(phone: str) -> Tuple[bool, str]:
    """
    Validate a phone number in E.164 format (e.g., +14155551234).

    Args:
        phone: The phone number to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not phone or not isinstance(phone, str):
        return False, "Phone number is required."

    phone = phone.strip()

    if not _PHONE_PATTERN.match(phone):
        return False, "Phone number must be in E.164 format (e.g., +14155551234)."

    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate password strength.

    Rules:
        - Minimum PASSWORD_MIN_LENGTH characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit

    Args:
        password: The password to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not password or not isinstance(password, str):
        return False, "Password is required."

    if len(password) < PASSWORD_MIN_LENGTH:
        return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit."

    return True, ""


# ============================================================================
# Password Hashing
# ============================================================================


def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt.

    Args:
        password: The plaintext password.

    Returns:
        The bcrypt hash as a UTF-8 string.

    Raises:
        RuntimeError: If bcrypt is not installed.
    """
    if not BCRYPT_AVAILABLE:
        raise RuntimeError("bcrypt is not installed. Install with: pip install bcrypt")

    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def check_password(password: str, password_hash: str) -> bool:
    """
    Verify a plaintext password against a bcrypt hash.

    Args:
        password: The plaintext password to check.
        password_hash: The bcrypt hash to check against.

    Returns:
        True if the password matches, False otherwise.

    Raises:
        RuntimeError: If bcrypt is not installed.
    """
    if not BCRYPT_AVAILABLE:
        raise RuntimeError("bcrypt is not installed. Install with: pip install bcrypt")

    try:
        password_bytes = password.encode("utf-8")
        hash_bytes = password_hash.encode("utf-8")
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception as e:
        logger.error("Password verification error: %s", type(e).__name__)
        return False


# ============================================================================
# User Repository — Abstract Base Class
# ============================================================================


class UserRepository(ABC):
    """
    Abstract base class defining the interface for user data access.

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
    def create_user(
        self,
        username: str,
        email: str,
        phone_number: str,
        password: str,
    ) -> Dict[str, Any]:
        """
        Create a new user account.

        Args:
            username: Unique username.
            email: Unique email address.
            phone_number: Phone number in E.164 format.
            password: Plaintext password (will be hashed before storage).

        Returns:
            Dict with keys: success (bool), error (str or None), user (dict or None).
        """
        ...

    @abstractmethod
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user document by username.

        Returns:
            User dict (without password_hash) or None if not found.
        """
        ...

    @abstractmethod
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user document by email.

        Returns:
            User dict (without password_hash) or None if not found.
        """
        ...

    @abstractmethod
    def update_user_profile(
        self, username: str, fields: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Update profile fields for the given user.

        Allowed fields: email, phone_number, notification_method,
        notifications_enabled.

        Args:
            username: The user to update.
            fields: Dict of field names to new values.

        Returns:
            Tuple of (success, error_message).
        """
        ...

    @abstractmethod
    def verify_password(self, username: str, password: str) -> bool:
        """
        Verify a plaintext password against the stored hash for the given user.

        Returns:
            True if credentials are valid, False otherwise.
        """
        ...

    @abstractmethod
    def delete_user(self, username: str) -> bool:
        """
        Delete a user account by username.

        Returns:
            True if the user was deleted, False otherwise.
        """
        ...


# ============================================================================
# Rate Limiter
# ============================================================================


class LoginRateLimiter:
    """
    Rate limiter for login attempts backed by a MongoDB collection with a TTL index.

    Records failed login attempts per username and blocks further attempts once
    the threshold is exceeded within the configured time window. The TTL index
    automatically removes expired attempt records.

    Graceful degradation: if MongoDB is unavailable the limiter logs a warning
    and allows the attempt (fail-open for availability).
    """

    def __init__(self, db=None):
        """
        Initialize the rate limiter.

        Args:
            db: A pymongo Database instance. If None, rate limiting is disabled.
        """
        self._collection = None
        if db is not None:
            try:
                self._collection = db[LOGIN_ATTEMPTS_COLLECTION_NAME]
                # Ensure TTL index — entries expire after the configured window
                self._collection.create_index(
                    "attempted_at",
                    expireAfterSeconds=LOGIN_ATTEMPT_WINDOW_MINUTES * 60,
                    name="ttl_login_attempts",
                )
                logger.info("Login rate limiter initialized with TTL index")
            except Exception as e:
                logger.warning("Could not initialize rate limiter collection: %s", e)
                self._collection = None

    def record_failed_attempt(self, username: str) -> None:
        """
        Record a failed login attempt for the given username.

        Args:
            username: The username that failed to log in.
        """
        if self._collection is None:
            return
        try:
            self._collection.insert_one({
                "username": username,
                "attempted_at": datetime.now(timezone.utc),
            })
        except Exception as e:
            logger.warning("Failed to record login attempt: %s", e)

    def is_rate_limited(self, username: str) -> bool:
        """
        Check whether the given username has exceeded the failed-login threshold.

        Args:
            username: The username to check.

        Returns:
            True if the user is rate-limited, False otherwise.
        """
        if self._collection is None:
            return False
        try:
            count = self._collection.count_documents({"username": username})
            return count >= MAX_LOGIN_ATTEMPTS
        except Exception as e:
            logger.warning("Rate limit check failed, allowing attempt: %s", e)
            return False

    def clear_attempts(self, username: str) -> None:
        """
        Clear all recorded failed attempts for a username (e.g., after a
        successful login).

        Args:
            username: The username whose attempts should be cleared.
        """
        if self._collection is None:
            return
        try:
            self._collection.delete_many({"username": username})
        except Exception as e:
            logger.warning("Failed to clear login attempts: %s", e)


# ============================================================================
# MongoDB User Repository
# ============================================================================

# Fields that may be updated via update_user_profile
_ALLOWED_PROFILE_FIELDS = {
    "email",
    "phone_number",
    "notification_method",
    "notifications_enabled",
}

# Fields that require validation when updated
_FIELD_VALIDATORS = {
    "email": validate_email,
    "phone_number": validate_phone,
}

# Valid notification_method values
_VALID_NOTIFICATION_METHODS = {"sms", "email", "both"}


class MongoUserRepository(UserRepository):
    """
    MongoDB-backed implementation of UserRepository.

    Stores user documents in the ``users`` collection of the ``bedside_dx``
    database. The document schema uses flat, normalized fields so it maps
    cleanly to a relational table for future Azure SQL migration.

    PII Protection:
        Phone numbers and emails are never logged, never included in error
        messages, and never exposed beyond the authenticated user's own
        profile view.
    """

    def __init__(
        self,
        uri: str = None,
        database_name: str = None,
        collection_name: str = None,
    ):
        """
        Initialize the MongoDB user repository.

        Args:
            uri: MongoDB connection string (defaults to config MONGODB_URI).
            database_name: Database name (defaults to config DATABASE_NAME).
            collection_name: Users collection name (defaults to config USERS_COLLECTION_NAME).
        """
        self.uri = uri or MONGODB_URI
        self.database_name = database_name or DATABASE_NAME
        self.collection_name = collection_name or USERS_COLLECTION_NAME
        self.client = None
        self.db = None
        self.collection = None
        self._connected = False
        self.rate_limiter: Optional[LoginRateLimiter] = None

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
            self.collection = self.db[self.collection_name]
            self._connected = True

            logger.info(
                "Connected to MongoDB users collection: %s.%s",
                self.database_name,
                self.collection_name,
            )

            self._ensure_indexes()

            # Initialize rate limiter with the same database
            self.rate_limiter = LoginRateLimiter(self.db)

            return True

        except ConnectionFailure as e:
            logger.error("Failed to connect to MongoDB for user auth: %s", e)
            return False
        except Exception as e:
            logger.error("MongoDB user auth connection error: %s", e)
            return False

    def _ensure_indexes(self) -> None:
        """Ensure required unique indexes exist on the users collection."""
        try:
            existing = self.collection.index_information()
            if "username_unique" not in existing:
                self.collection.create_index(
                    [("username", ASCENDING)],
                    unique=True,
                    name="username_unique",
                )
                logger.info("Created unique index on username")
            if "email_unique" not in existing:
                self.collection.create_index(
                    [("email", ASCENDING)],
                    unique=True,
                    name="email_unique",
                )
                logger.info("Created unique index on email")
        except OperationFailure as e:
            logger.warning("Could not create user indexes (may already exist): %s", e)
        except Exception as e:
            logger.warning("User index creation warning: %s", e)

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("MongoDB user auth connection closed")

    def is_connected(self) -> bool:
        """Check if the repository is connected."""
        return self._connected and self.collection is not None

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def create_user(
        self,
        username: str,
        email: str,
        phone_number: str,
        password: str,
    ) -> Dict[str, Any]:
        """
        Create a new user account.

        Validates all inputs, hashes the password, and inserts a flat document.

        Args:
            username: Unique username.
            email: Unique email address.
            phone_number: Phone number in E.164 format.
            password: Plaintext password (will be hashed).

        Returns:
            Dict with keys: success (bool), error (str or None), user (dict or None).
        """
        # --- Input validation ---
        valid, msg = validate_username(username)
        if not valid:
            return {"success": False, "error": msg, "user": None}

        valid, msg = validate_email(email)
        if not valid:
            return {"success": False, "error": msg, "user": None}

        valid, msg = validate_phone(phone_number)
        if not valid:
            return {"success": False, "error": msg, "user": None}

        valid, msg = validate_password(password)
        if not valid:
            return {"success": False, "error": msg, "user": None}

        if not self.is_connected():
            return {"success": False, "error": "Database not connected.", "user": None}

        # --- Build document ---
        now = datetime.now(timezone.utc)
        user_doc = {
            "username": username.strip(),
            "email": email.strip().lower(),
            "phone_number": phone_number.strip(),
            "password_hash": hash_password(password),
            "notification_method": "email",  # default placeholder
            "notifications_enabled": False,  # placeholder
            "created_at": now,
            "updated_at": now,
        }

        try:
            self.collection.insert_one(user_doc)
            logger.info("User created: %s", username)
            # Return user dict WITHOUT password_hash
            safe_user = {k: v for k, v in user_doc.items() if k not in ("password_hash", "_id")}
            return {"success": True, "error": None, "user": safe_user}

        except DuplicateKeyError as e:
            error_str = str(e).lower()
            if "email" in error_str:
                logger.warning("Duplicate email during registration for user: %s", username)
                return {"success": False, "error": "An account with this email already exists.", "user": None}
            else:
                logger.warning("Duplicate username during registration: %s", username)
                return {"success": False, "error": "This username is already taken.", "user": None}
        except Exception as e:
            logger.error("Failed to create user %s: %s", username, type(e).__name__)
            return {"success": False, "error": "Account creation failed. Please try again.", "user": None}

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user by username.

        Returns:
            User dict (without password_hash and _id) or None.
        """
        if not self.is_connected():
            return None
        try:
            doc = self.collection.find_one(
                {"username": username},
                {"_id": 0, "password_hash": 0},
            )
            return doc
        except Exception as e:
            logger.error("Failed to get user by username: %s", type(e).__name__)
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user by email address.

        Returns:
            User dict (without password_hash and _id) or None.
        """
        if not self.is_connected():
            return None
        try:
            doc = self.collection.find_one(
                {"email": email.strip().lower()},
                {"_id": 0, "password_hash": 0},
            )
            return doc
        except Exception as e:
            logger.error("Failed to get user by email: %s", type(e).__name__)
            return None

    def update_user_profile(
        self, username: str, fields: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Update allowed profile fields for the given user.

        Allowed fields: email, phone_number, notification_method,
        notifications_enabled.

        Args:
            username: The username to update.
            fields: Dict of field names to new values.

        Returns:
            Tuple of (success, error_message).
        """
        if not self.is_connected():
            return False, "Database not connected."

        if not fields:
            return False, "No fields to update."

        # Filter to allowed fields
        update_fields = {}
        for key, value in fields.items():
            if key not in _ALLOWED_PROFILE_FIELDS:
                return False, f"Field '{key}' cannot be updated."

            # Validate fields that have validators
            if key in _FIELD_VALIDATORS:
                valid, msg = _FIELD_VALIDATORS[key](value)
                if not valid:
                    return False, msg

            # Validate notification_method enum
            if key == "notification_method" and value not in _VALID_NOTIFICATION_METHODS:
                return False, f"notification_method must be one of: {', '.join(_VALID_NOTIFICATION_METHODS)}"

            # Validate notifications_enabled type
            if key == "notifications_enabled" and not isinstance(value, bool):
                return False, "notifications_enabled must be a boolean."

            # Normalize email to lowercase
            if key == "email":
                value = value.strip().lower()

            update_fields[key] = value

        update_fields["updated_at"] = datetime.now(timezone.utc)

        try:
            result = self.collection.update_one(
                {"username": username},
                {"$set": update_fields},
            )
            if result.matched_count == 0:
                return False, "User not found."
            logger.info("Profile updated for user: %s", username)
            return True, ""
        except DuplicateKeyError:
            return False, "This email is already in use by another account."
        except Exception as e:
            logger.error("Failed to update profile for %s: %s", username, type(e).__name__)
            return False, "Profile update failed. Please try again."

    def verify_password(self, username: str, password: str) -> bool:
        """
        Verify a plaintext password against the stored hash.

        Args:
            username: The username to authenticate.
            password: The plaintext password.

        Returns:
            True if credentials are valid, False otherwise.
        """
        if not self.is_connected():
            return False
        try:
            doc = self.collection.find_one(
                {"username": username},
                {"password_hash": 1, "_id": 0},
            )
            if not doc or "password_hash" not in doc:
                return False
            return check_password(password, doc["password_hash"])
        except Exception as e:
            logger.error("Password verification failed for %s: %s", username, type(e).__name__)
            return False

    def delete_user(self, username: str) -> bool:
        """
        Delete a user account by username.

        Args:
            username: The username to delete.

        Returns:
            True if the user was deleted, False otherwise.
        """
        if not self.is_connected():
            return False
        try:
            result = self.collection.delete_one({"username": username})
            if result.deleted_count > 0:
                logger.info("User deleted: %s", username)
                # Also clear any rate-limit records
                if self.rate_limiter:
                    self.rate_limiter.clear_attempts(username)
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete user %s: %s", username, type(e).__name__)
            return False

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------

    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate a user with rate limiting.

        Checks rate limits before attempting password verification. On failure,
        records the attempt. On success, clears the attempt history.

        Args:
            username: The username to authenticate.
            password: The plaintext password.

        Returns:
            Dict with keys: success (bool), error (str or None), user (dict or None).
        """
        if not self.is_connected():
            return {"success": False, "error": "Database not connected.", "user": None}

        # Check rate limit
        if self.rate_limiter and self.rate_limiter.is_rate_limited(username):
            logger.warning("Rate-limited login attempt for user: %s", username)
            return {
                "success": False,
                "error": f"Too many failed login attempts. Please try again in {LOGIN_ATTEMPT_WINDOW_MINUTES} minutes.",
                "user": None,
            }

        # Verify credentials
        if not self.verify_password(username, password):
            # Record failed attempt
            if self.rate_limiter:
                self.rate_limiter.record_failed_attempt(username)
            return {"success": False, "error": "Invalid username or password.", "user": None}

        # Success — clear failed attempts
        if self.rate_limiter:
            self.rate_limiter.clear_attempts(username)

        user = self.get_user_by_username(username)
        logger.info("User authenticated: %s", username)
        return {"success": True, "error": None, "user": user}
