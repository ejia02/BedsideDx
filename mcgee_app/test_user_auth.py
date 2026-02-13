#!/usr/bin/env python3
"""
Test script for user authentication system.

This script verifies:
1. Input validation (username, email, phone, password)
2. Password hashing correctness
3. User creation (happy path)
4. Duplicate username/email prevention
5. Login / password verification (correct + incorrect)
6. Profile updates
7. Rate limiting (5 failed logins triggers block)
8. Cleanup: deletes test user at end

Usage:
    python mcgee_app/test_user_auth.py
"""

import sys

from config import MONGODB_URI, DATABASE_NAME, USERS_COLLECTION_NAME

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

try:
    from user_auth import (
        validate_username,
        validate_email,
        validate_phone,
        validate_password,
        hash_password,
        check_password,
        MongoUserRepository,
        BCRYPT_AVAILABLE,
        PYMONGO_AVAILABLE,
    )
except ImportError as e:
    print(f"❌ Failed to import user_auth module: {e}")
    sys.exit(1)


# Test user constants — NOT real PII
TEST_USERNAME = "_test_user_auth_script_"
TEST_EMAIL = "testuser_auth_script@example.com"
TEST_PHONE = "+10000000000"
TEST_PASSWORD = "TestPass1"

TEST_USERNAME_2 = "_test_user_auth_script_2_"
TEST_EMAIL_2 = "testuser_auth_script_2@example.com"

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
# 1. Input Validation
# ===================================================================

def test_input_validation():
    """Test all validation functions with valid and invalid inputs."""
    print()
    print("1️⃣  Input Validation")
    print("-" * 50)

    # --- Username ---
    ok, _ = validate_username("alice")
    check("Valid username accepted", ok)

    ok, _ = validate_username("ab")
    check("Too-short username rejected", not ok)

    ok, _ = validate_username("a" * 31)
    check("Too-long username rejected", not ok)

    ok, _ = validate_username("bad user!")
    check("Username with special chars rejected", not ok)

    ok, _ = validate_username("")
    check("Empty username rejected", not ok)

    # --- Email ---
    ok, _ = validate_email("user@example.com")
    check("Valid email accepted", ok)

    ok, _ = validate_email("not-an-email")
    check("Invalid email rejected", not ok)

    ok, _ = validate_email("")
    check("Empty email rejected", not ok)

    # --- Phone ---
    ok, _ = validate_phone("+14155551234")
    check("Valid E.164 phone accepted", ok)

    ok, _ = validate_phone("4155551234")
    check("Phone without + rejected", not ok)

    ok, _ = validate_phone("+1")
    check("Too-short phone rejected", not ok)

    ok, _ = validate_phone("")
    check("Empty phone rejected", not ok)

    # --- Password ---
    ok, _ = validate_password("GoodPass1")
    check("Valid password accepted", ok)

    ok, _ = validate_password("short1A")
    check("Short password rejected", not ok)

    ok, _ = validate_password("nouppercase1")
    check("Password without uppercase rejected", not ok)

    ok, _ = validate_password("NOLOWERCASE1")
    check("Password without lowercase rejected", not ok)

    ok, _ = validate_password("NoDigitsHere")
    check("Password without digit rejected", not ok)

    ok, _ = validate_password("")
    check("Empty password rejected", not ok)


# ===================================================================
# 2. Password Hashing
# ===================================================================

def test_password_hashing():
    """Test bcrypt hashing and verification."""
    print()
    print("2️⃣  Password Hashing")
    print("-" * 50)

    if not BCRYPT_AVAILABLE:
        print("   ⚠️  bcrypt not installed — skipping hashing tests")
        return

    hashed = hash_password("MyPassword1")
    check("hash_password returns a string", isinstance(hashed, str))
    check("Hash is not plaintext", hashed != "MyPassword1")
    check("Hash starts with bcrypt prefix", hashed.startswith("$2"))

    check("Correct password verifies", check_password("MyPassword1", hashed))
    check("Wrong password does not verify", not check_password("WrongPassword1", hashed))

    # Two hashes of the same password should differ (different salts)
    hashed2 = hash_password("MyPassword1")
    check("Same password produces different hashes (salting)", hashed != hashed2)
    check("Both hashes verify the same password",
          check_password("MyPassword1", hashed) and check_password("MyPassword1", hashed2))


# ===================================================================
# 3-7. Database-dependent tests
# ===================================================================

def test_database_operations():
    """Test user CRUD, duplicate prevention, auth, profile update, and rate limiting."""
    print()
    print("3️⃣  User Creation")
    print("-" * 50)

    if not PYMONGO_AVAILABLE:
        print("   ⚠️  pymongo not installed — skipping database tests")
        return

    if not BCRYPT_AVAILABLE:
        print("   ⚠️  bcrypt not installed — skipping database tests")
        return

    repo = MongoUserRepository()
    if not repo.connect():
        print("   ⚠️  Could not connect to MongoDB — skipping database tests")
        print(f"      URI: {MONGODB_URI}")
        return

    # --- Cleanup any leftover test data ---
    repo.delete_user(TEST_USERNAME)
    repo.delete_user(TEST_USERNAME_2)

    # --- 3. Create user ---
    result = repo.create_user(TEST_USERNAME, TEST_EMAIL, TEST_PHONE, TEST_PASSWORD)
    check("User creation succeeds", result["success"], result.get("error", ""))
    check("Returned user dict has username", result.get("user", {}).get("username") == TEST_USERNAME)
    check("Returned user dict has NO password_hash", "password_hash" not in (result.get("user") or {}))

    # --- 4. Duplicate prevention ---
    print()
    print("4️⃣  Duplicate Prevention")
    print("-" * 50)

    dup_result = repo.create_user(TEST_USERNAME, "other@example.com", "+19999999999", "OtherPass1")
    check("Duplicate username rejected", not dup_result["success"])
    check("Error mentions username", "username" in (dup_result.get("error") or "").lower())

    dup_result2 = repo.create_user("unique_user_xyz", TEST_EMAIL, "+19999999999", "OtherPass1")
    check("Duplicate email rejected", not dup_result2["success"])
    check("Error mentions email", "email" in (dup_result2.get("error") or "").lower())

    # --- 5. Login / password verification ---
    print()
    print("5️⃣  Login & Password Verification")
    print("-" * 50)

    check("verify_password with correct password", repo.verify_password(TEST_USERNAME, TEST_PASSWORD))
    check("verify_password with wrong password", not repo.verify_password(TEST_USERNAME, "WrongPass1"))
    check("verify_password for non-existent user", not repo.verify_password("no_such_user", TEST_PASSWORD))

    auth_result = repo.authenticate(TEST_USERNAME, TEST_PASSWORD)
    check("authenticate succeeds with correct credentials", auth_result["success"])
    check("authenticate returns user dict", auth_result.get("user") is not None)

    auth_fail = repo.authenticate(TEST_USERNAME, "WrongPass1")
    check("authenticate fails with wrong password", not auth_fail["success"])

    # --- 6. Profile updates ---
    print()
    print("6️⃣  Profile Updates")
    print("-" * 50)

    ok, err = repo.update_user_profile(TEST_USERNAME, {"email": "updated@example.com"})
    check("Email update succeeds", ok, err)

    user = repo.get_user_by_username(TEST_USERNAME)
    check("Email was actually updated", user is not None and user.get("email") == "updated@example.com")

    ok, err = repo.update_user_profile(TEST_USERNAME, {"phone_number": "+12025551234"})
    check("Phone update succeeds", ok, err)

    ok, err = repo.update_user_profile(TEST_USERNAME, {"notification_method": "sms"})
    check("Notification method update succeeds", ok, err)

    ok, err = repo.update_user_profile(TEST_USERNAME, {"email": "not-an-email"})
    check("Invalid email update rejected", not ok)

    ok, err = repo.update_user_profile(TEST_USERNAME, {"phone_number": "12345"})
    check("Invalid phone update rejected", not ok)

    ok, err = repo.update_user_profile(TEST_USERNAME, {"password_hash": "sneaky"})
    check("Disallowed field update rejected", not ok)

    ok, err = repo.update_user_profile("no_such_user_xyz", {"email": "x@example.com"})
    check("Update for non-existent user fails", not ok)

    # --- 7. Rate limiting ---
    print()
    print("7️⃣  Rate Limiting")
    print("-" * 50)

    # Use a second test user so rate-limit state doesn't interfere
    repo.create_user(TEST_USERNAME_2, TEST_EMAIL_2, "+10000000001", TEST_PASSWORD)

    for i in range(5):
        repo.authenticate(TEST_USERNAME_2, "WrongPassword1")

    blocked = repo.authenticate(TEST_USERNAME_2, TEST_PASSWORD)
    check("Rate limited after 5 failed attempts", not blocked["success"])
    check("Error mentions rate limit",
          "too many" in (blocked.get("error") or "").lower()
          or "try again" in (blocked.get("error") or "").lower())

    # --- Cleanup ---
    print()
    print("8️⃣  Cleanup")
    print("-" * 50)

    check("Delete test user", repo.delete_user(TEST_USERNAME))
    check("Delete second test user", repo.delete_user(TEST_USERNAME_2))
    check("Deleted user no longer found", repo.get_user_by_username(TEST_USERNAME) is None)

    repo.close()


# ===================================================================
# Main
# ===================================================================

def main():
    """Run all tests."""
    print()
    print("🩺 User Authentication Test Suite")
    print("=" * 50)
    print(f"URI: {MONGODB_URI}")
    print(f"Database: {DATABASE_NAME}")
    print(f"Users collection: {USERS_COLLECTION_NAME}")

    test_input_validation()
    test_password_hashing()
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
