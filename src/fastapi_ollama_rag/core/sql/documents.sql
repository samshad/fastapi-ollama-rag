-- name: check_file_exists(file_hash, user_id)^
-- Checks if a specific user has already uploaded this file hash.
SELECT 1 FROM files 
WHERE file_hash = :file_hash AND user_id = :user_id 
LIMIT 1;

-- name: create_file_record(user_id, filename, file_hash)^
-- Inserts a file record and returns its UUID.
INSERT INTO files (user_id, filename, file_hash) 
VALUES (:user_id, :filename, :file_hash) 
RETURNING id;

-- name: get_bulk_insert_sql()#
-- Returns the raw SQL string for bulk chunk insertion.
INSERT INTO documents (content, metadata, embedding, user_id, file_id) 
VALUES ($1, $2::jsonb, $3::vector, $4, $5);

-- name: search_vectors(embedding, user_id, limit_val)
-- Vector similarity search locked to the specific user. Notice NO symbol at the end!
SELECT content, metadata, 1 - (embedding <=> :embedding::vector) AS similarity
FROM documents
WHERE user_id = :user_id
ORDER BY embedding <=> :embedding::vector
LIMIT :limit_val;

-- name: get_user_files(user_id)
-- Retrieves metadata for all files owned by the authenticated user.
SELECT id, filename, file_hash, created_at
FROM files
WHERE user_id = :user_id
ORDER BY created_at DESC;

-- name: delete_file_chunks(file_id, user_id)!
-- Deletes all vector chunks associated with a file.
DELETE FROM documents
WHERE file_id = :file_id AND user_id = :user_id;

-- name: delete_file_record(file_id, user_id)!
-- Deletes the actual file record.
DELETE FROM files
WHERE id = :file_id AND user_id = :user_id;
