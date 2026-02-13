#!/usr/bin/env python3
"""
Test script for chat history persistence system.

This script verifies:
1. Conversation CRUD (create, get, list, update title, delete)
2. Message insertion and chronological retrieval ordering
3. Cascade delete (deleting a conversation removes all its messages)
4. User isolation (user A cannot see user B's conversations)
5. Serialization round-trip for all message types (user, differential, strategy, error)
6. Title sanitization (HTML tags stripped, long titles truncated)
7. Cleanup: deletes all test data at end

Usage:
    python mcgee_app/test_chat_history.py
"""

import sys
import time
import json

from config import MONGODB_URI, DATABASE_NAME, CONVERSATIONS_COLLECTION_NAME, MESSAGES_COLLECTION_NAME

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

try:
    from chat_history import (
        MongoChatHistoryRepository,
        serialize_message,
        deserialize_message,
        generate_conversation_title,
        sanitize_html,
        PYMONGO_AVAILABLE,
    )
except ImportError as e:
    print(f"❌ Failed to import chat_history module: {e}")
    sys.exit(1)


# Test user constants — NOT real users
TEST_USER_A = "_test_chat_user_a_"
TEST_USER_B = "_test_chat_user_b_"

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = ""):
    """Record a pass/fail for a single check."""
    global passed, failed
    if condition:
        passed += 1
        print(f"   ✅ {label}")
    else:
        failed += 1
        msg = f"   ❌ {label}"
        if detail:
            msg += f" — {detail}"
        print(msg)


# ===================================================================
# 1. Title Sanitization & Generation
# ===================================================================

def test_title_sanitization():
    """Test HTML sanitization and title generation utilities."""
    print()
    print("1️⃣  Title Sanitization & Generation")
    print("-" * 50)

    # HTML stripping
    check(
        "HTML tags stripped from title",
        sanitize_html("<script>alert('xss')</script>Hello") == "alert('xss')Hello",
    )
    check(
        "Nested HTML tags stripped",
        sanitize_html("<b><i>Bold Italic</i></b>") == "Bold Italic",
    )
    check(
        "Plain text unchanged",
        sanitize_html("No tags here") == "No tags here",
    )
    check(
        "Empty string returns empty",
        sanitize_html("") == "",
    )
    check(
        "None returns empty",
        sanitize_html(None) == "",
    )

    # Title generation
    check(
        "Short query becomes title directly",
        generate_conversation_title("Chest pain evaluation") == "Chest pain evaluation",
    )

    long_query = "A" * 200
    title = generate_conversation_title(long_query)
    check(
        "Long query truncated to max length",
        len(title) <= 83,  # 80 + "..."
        f"Length was {len(title)}",
    )

    check(
        "HTML stripped in title generation",
        "<" not in generate_conversation_title("<b>Bold</b> query text here"),
    )

    check(
        "Empty query fallback",
        generate_conversation_title("") == "New Conversation",
    )

    check(
        "None query fallback",
        generate_conversation_title(None) == "New Conversation",
    )


# ===================================================================
# 2. Serialization Round-Trip
# ===================================================================

def test_serialization_roundtrip():
    """Test serialize then deserialize for all message types."""
    print()
    print("2️⃣  Serialization Round-Trip")
    print("-" * 50)

    # --- User message ---
    user_msg = {"role": "user", "content": "Patient with chest pain"}
    msg_type, text, json_data = serialize_message(user_msg)
    check("User message type is 'query'", msg_type == "query")
    check("User message content preserved", text == "Patient with chest pain")
    check("User message has no content_json", json_data is None)

    reconstructed = deserialize_message({
        "role": "user",
        "message_type": msg_type,
        "content_text": text,
        "content_json": json_data,
    })
    check("User message round-trip: role", reconstructed["role"] == "user")
    check("User message round-trip: content", reconstructed["content"] == "Patient with chest pain")

    # --- Differential message ---
    diff_items = [
        {"name": "Pneumonia", "rationale": "Productive cough and fever"},
        {"name": "PE", "rationale": "Sudden onset dyspnea"},
    ]
    diff_msg = {
        "role": "assistant",
        "type": "differential",
        "content": "**Differential diagnosis**\n- Pneumonia\n- PE",
        "items": diff_items,
    }
    msg_type, text, json_data = serialize_message(diff_msg)
    check("Differential type is 'differential'", msg_type == "differential")
    check("Differential content_text preserved", "Pneumonia" in text)
    check("Differential content_json is not None", json_data is not None)

    reconstructed = deserialize_message({
        "role": "assistant",
        "message_type": msg_type,
        "content_text": text,
        "content_json": json_data,
    })
    check("Differential round-trip: type", reconstructed.get("type") == "differential")
    check("Differential round-trip: items count", len(reconstructed.get("items", [])) == 2)
    check(
        "Differential round-trip: item name preserved",
        reconstructed.get("items", [{}])[0].get("name") == "Pneumonia",
    )

    # --- Strategy message ---
    strategy_result = {
        "success": True,
        "strategy": "Perform auscultation...",
        "strategy_structured": {
            "sections": [
                {
                    "system": "Respiratory",
                    "diseases": [
                        {
                            "name": "Pneumonia",
                            "rule_in": [{"name": "Crackles", "lr_positive": "3.5"}],
                            "rule_out": [],
                        }
                    ],
                }
            ]
        },
        "processing_time": 4.2,
        "evidence": [{"id": 1}, {"id": 2}, {"id": 3}],
        "categories": {
            "high_yield_positive": [{"x": 1}],
            "high_yield_negative": [{"x": 2}],
        },
    }
    strategy_msg = {"role": "assistant", "type": "strategy", "result": strategy_result}
    msg_type, text, json_data = serialize_message(strategy_msg)
    check("Strategy type is 'strategy'", msg_type == "strategy")
    check("Strategy content_text has readable text", len(text) > 0)
    check("Strategy content_json is not None", json_data is not None)

    # Verify JSON payload is minimal (no raw evidence/categories)
    payload = json.loads(json_data)
    check("Strategy JSON has strategy_structured", "strategy_structured" in payload)
    check("Strategy JSON has processing_time", payload.get("processing_time") == 4.2)
    check("Strategy JSON has evidence_count", payload.get("evidence_count") == 3)
    check("Strategy JSON has high_yield_count", payload.get("high_yield_count") == 2)
    check("Strategy JSON does NOT have raw evidence", "evidence" not in payload)
    check("Strategy JSON does NOT have categories", "categories" not in payload)

    reconstructed = deserialize_message({
        "role": "assistant",
        "message_type": msg_type,
        "content_text": text,
        "content_json": json_data,
    })
    check("Strategy round-trip: type", reconstructed.get("type") == "strategy")
    r = reconstructed.get("result", {})
    check(
        "Strategy round-trip: strategy_structured sections",
        len(r.get("strategy_structured", {}).get("sections", [])) == 1,
    )
    check(
        "Strategy round-trip: processing_time",
        r.get("processing_time") == 4.2,
    )

    # --- Error message ---
    error_msg = {"role": "assistant", "type": "error", "content": "Something broke"}
    msg_type, text, json_data = serialize_message(error_msg)
    check("Error type is 'error'", msg_type == "error")
    check("Error content preserved", text == "Something broke")
    check("Error has no content_json", json_data is None)

    reconstructed = deserialize_message({
        "role": "assistant",
        "message_type": msg_type,
        "content_text": text,
        "content_json": json_data,
    })
    check("Error round-trip: type", reconstructed.get("type") == "error")
    check("Error round-trip: content", reconstructed.get("content") == "Something broke")

    # --- Corrupt JSON graceful fallback ---
    reconstructed = deserialize_message({
        "role": "assistant",
        "message_type": "strategy",
        "content_text": "Fallback text here",
        "content_json": "NOT VALID JSON{{{",
    })
    check(
        "Corrupt strategy JSON falls back gracefully",
        reconstructed.get("content") == "Fallback text here",
    )


# ===================================================================
# 3-7. Database-dependent tests
# ===================================================================

def test_database_operations():
    """Test conversation CRUD, messages, cascade delete, user isolation."""
    print()
    print("3️⃣  Conversation CRUD")
    print("-" * 50)

    if not PYMONGO_AVAILABLE:
        print("   ⚠️  pymongo not installed — skipping database tests")
        return

    repo = MongoChatHistoryRepository()
    if not repo.connect():
        print("   ⚠️  Could not connect to MongoDB — skipping database tests")
        print(f"      URI: {MONGODB_URI}")
        return

    # --- Cleanup any leftover test data ---
    repo.delete_all_conversations_for_user(TEST_USER_A)
    repo.delete_all_conversations_for_user(TEST_USER_B)

    # --- 3a. Create conversation ---
    conv = repo.create_conversation(TEST_USER_A, "Test conversation one")
    check("Conversation created", conv is not None)
    check("Conversation has conversation_id", "conversation_id" in (conv or {}))
    check("Conversation has title", (conv or {}).get("title") == "Test conversation one")
    check("Conversation has username", (conv or {}).get("username") == TEST_USER_A)

    conv_id = conv["conversation_id"]

    # --- 3b. Get conversation ---
    fetched = repo.get_conversation(conv_id, TEST_USER_A)
    check("Get conversation returns data", fetched is not None)
    check("Get conversation title matches", (fetched or {}).get("title") == "Test conversation one")

    # --- 3c. List conversations ---
    conv2 = repo.create_conversation(TEST_USER_A, "Test conversation two")
    conv2_id = conv2["conversation_id"]

    convs = repo.get_conversations_for_user(TEST_USER_A)
    check("List returns 2 conversations", len(convs) == 2)
    # Most recent first (conv2 was created after conv1)
    check(
        "List is sorted by updated_at desc (newest first)",
        convs[0]["conversation_id"] == conv2_id,
    )

    # --- 3d. Update title ---
    print()
    print("4️⃣  Update Conversation Title")
    print("-" * 50)

    ok = repo.update_conversation_title(conv_id, TEST_USER_A, "Updated title")
    check("Title update succeeds", ok)

    fetched = repo.get_conversation(conv_id, TEST_USER_A)
    check("Title was actually updated", (fetched or {}).get("title") == "Updated title")

    # HTML in title should be stripped
    ok = repo.update_conversation_title(conv_id, TEST_USER_A, "<b>Bold</b> title")
    fetched = repo.get_conversation(conv_id, TEST_USER_A)
    check("HTML stripped from updated title", (fetched or {}).get("title") == "Bold title")

    # --- 5. Messages ---
    print()
    print("5️⃣  Message Insertion & Ordering")
    print("-" * 50)

    msg1 = repo.add_message(conv_id, TEST_USER_A, "user", "query", "What is DVT?")
    check("Message 1 inserted", msg1 is not None)
    check("Message 1 has message_id", "message_id" in (msg1 or {}))

    time.sleep(0.05)  # Ensure different timestamps

    msg2 = repo.add_message(
        conv_id,
        TEST_USER_A,
        "assistant",
        "differential",
        "**Differential**\n- DVT\n- Cellulitis",
        json.dumps([{"name": "DVT"}, {"name": "Cellulitis"}]),
    )
    check("Message 2 inserted", msg2 is not None)

    time.sleep(0.05)

    msg3 = repo.add_message(
        conv_id,
        TEST_USER_A,
        "assistant",
        "strategy",
        "Check Homan's sign...",
        json.dumps({"strategy_structured": {"sections": []}, "processing_time": 1.5}),
    )
    check("Message 3 inserted", msg3 is not None)

    messages = repo.get_messages(conv_id, TEST_USER_A)
    check("Retrieved 3 messages", len(messages) == 3)
    check(
        "Messages in chronological order (first is user query)",
        messages[0]["role"] == "user" and messages[0]["message_type"] == "query",
    )
    check(
        "Second message is differential",
        messages[1]["message_type"] == "differential",
    )
    check(
        "Third message is strategy",
        messages[2]["message_type"] == "strategy",
    )

    # --- 6. User isolation ---
    print()
    print("6️⃣  User Isolation")
    print("-" * 50)

    conv_b = repo.create_conversation(TEST_USER_B, "User B conversation")
    conv_b_id = conv_b["conversation_id"]
    repo.add_message(conv_b_id, TEST_USER_B, "user", "query", "Secret query")

    # User A should not see user B's conversations
    convs_a = repo.get_conversations_for_user(TEST_USER_A)
    conv_ids_a = {c["conversation_id"] for c in convs_a}
    check("User A cannot list user B's conversations", conv_b_id not in conv_ids_a)

    # User A should not be able to get user B's conversation
    fetched = repo.get_conversation(conv_b_id, TEST_USER_A)
    check("User A cannot get user B's conversation", fetched is None)

    # User A should not be able to get user B's messages
    msgs = repo.get_messages(conv_b_id, TEST_USER_A)
    check("User A cannot get user B's messages", len(msgs) == 0)

    # User A should not be able to add messages to user B's conversation
    result = repo.add_message(conv_b_id, TEST_USER_A, "user", "query", "Injected!")
    check("User A cannot add message to user B's conversation", result is None)

    # User A should not be able to delete user B's conversation
    ok = repo.delete_conversation(conv_b_id, TEST_USER_A)
    check("User A cannot delete user B's conversation", not ok)

    # User A should not be able to update user B's conversation title
    ok = repo.update_conversation_title(conv_b_id, TEST_USER_A, "Hacked title")
    check("User A cannot update user B's conversation title", not ok)

    # --- 7. Cascade delete ---
    print()
    print("7️⃣  Cascade Delete")
    print("-" * 50)

    # Delete conv_id (which has 3 messages)
    ok = repo.delete_conversation(conv_id, TEST_USER_A)
    check("Conversation deleted", ok)

    # Messages should also be gone
    messages = repo.get_messages(conv_id, TEST_USER_A)
    check("Messages cascade-deleted (0 remaining)", len(messages) == 0)

    # Conversation itself should be gone
    fetched = repo.get_conversation(conv_id, TEST_USER_A)
    check("Conversation no longer found after delete", fetched is None)

    # Delete all for user A (conv2 still exists)
    count = repo.delete_all_conversations_for_user(TEST_USER_A)
    check("delete_all_conversations_for_user returns correct count", count >= 1)

    convs_a = repo.get_conversations_for_user(TEST_USER_A)
    check("User A has 0 conversations after delete_all", len(convs_a) == 0)

    # --- Cleanup ---
    print()
    print("8️⃣  Cleanup")
    print("-" * 50)

    repo.delete_all_conversations_for_user(TEST_USER_A)
    repo.delete_all_conversations_for_user(TEST_USER_B)

    check(
        "User A cleanup complete",
        len(repo.get_conversations_for_user(TEST_USER_A)) == 0,
    )
    check(
        "User B cleanup complete",
        len(repo.get_conversations_for_user(TEST_USER_B)) == 0,
    )

    repo.close()


# ===================================================================
# Main
# ===================================================================

def main():
    """Run all tests."""
    print()
    print("🩺 Chat History Test Suite")
    print("=" * 50)
    print(f"URI: {MONGODB_URI}")
    print(f"Database: {DATABASE_NAME}")
    print(f"Conversations collection: {CONVERSATIONS_COLLECTION_NAME}")
    print(f"Messages collection: {MESSAGES_COLLECTION_NAME}")

    test_title_sanitization()
    test_serialization_roundtrip()
    test_database_operations()

    print()
    print("=" * 50)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
    print()

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
