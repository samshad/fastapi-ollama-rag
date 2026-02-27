INSERT INTO documents (content, metadata, embedding, user_id, file_id)
VALUES ($1, $2::jsonb, $3::vector, $4, $5);