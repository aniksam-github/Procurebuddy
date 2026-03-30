ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS username VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_base64 TEXT;

UPDATE users
SET display_name = email
WHERE display_name IS NULL OR BTRIM(display_name) = '';

UPDATE users
SET username = split_part(email, '@', 1)
WHERE username IS NULL OR BTRIM(username) = '';

ALTER TABLE users ALTER COLUMN display_name SET NOT NULL;
ALTER TABLE users ALTER COLUMN username SET NOT NULL;
