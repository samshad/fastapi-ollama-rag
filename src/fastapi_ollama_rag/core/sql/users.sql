-- name: get_user_by_email(email)^
-- Fetches a user by email.
SELECT id, email, hashed_password, is_verified
FROM users
WHERE email = :email;

-- name: create_user(email, hashed_password)^
-- Inserts a new verified user and returns the record.
INSERT INTO users (email, hashed_password, is_verified)
VALUES (:email, :hashed_password, TRUE)
RETURNING id, email;

-- name: save_otp(email, code, expires_at)!
-- Saves a generated OTP to the database.
INSERT INTO otps (email, code, expires_at)
VALUES (:email, :code, :expires_at);

-- name: get_valid_otp(email, code)^
-- Retrieves an OTP only if it matches and hasn't expired.
SELECT id FROM otps
WHERE email = :email AND code = :code AND expires_at > NOW()
ORDER BY created_at DESC LIMIT 1;

-- name: delete_otps_for_email(email)!
-- Cleans up OTPs after successful registration to prevent replay attacks.
DELETE FROM otps WHERE email = :email;

-- name: update_user_password(email, hashed_password)!
-- Updates an existing user's password.
UPDATE users
SET hashed_password = :hashed_password
WHERE email = :email;

-- name: get_user_by_id(user_id)^
-- Fetches a user by their UUID.
SELECT id, email, is_verified
FROM users
WHERE id = :user_id;
