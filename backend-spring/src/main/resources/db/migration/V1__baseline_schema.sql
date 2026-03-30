CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    display_name VARCHAR(255),
    username VARCHAR(255),
    avatar_base64 TEXT,
    password_hash VARCHAR(255) NOT NULL,
    must_change BOOLEAN NOT NULL,
    totp_enabled BOOLEAN NOT NULL,
    totp_secret VARCHAR(255),
    pending_totp_secret VARCHAR(255),
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_users_email ON users(email);

CREATE TABLE pending_otps (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    otp VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL
);

CREATE TABLE folders (
    id VARCHAR(255) PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_folders_user_created ON folders(user_id, created_at);

CREATE TABLE chats (
    id VARCHAR(255) PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id),
    folder_id VARCHAR(255) REFERENCES folders(id),
    title VARCHAR(255) NOT NULL,
    preview VARCHAR(500),
    pinned BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_chats_user_updated ON chats(user_id, updated_at);
CREATE INDEX idx_chats_folder ON chats(folder_id);
CREATE INDEX idx_chats_pinned_updated ON chats(pinned, updated_at);

CREATE TABLE messages (
    id BIGSERIAL PRIMARY KEY,
    chat_id VARCHAR(255) NOT NULL REFERENCES chats(id),
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL
);

CREATE INDEX idx_messages_chat_timestamp ON messages(chat_id, timestamp);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);

CREATE TABLE message_revisions (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT NOT NULL REFERENCES messages(id),
    source VARCHAR(32) NOT NULL,
    response TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_message_revisions_message_created ON message_revisions(message_id, created_at);

CREATE TABLE feedback (
    id BIGSERIAL PRIMARY KEY,
    message_id VARCHAR(255) NOT NULL,
    chat_id VARCHAR(255),
    user_email VARCHAR(255) NOT NULL,
    type VARCHAR(16) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    CONSTRAINT uk_feedback_user_message UNIQUE (user_email, message_id)
);

CREATE INDEX idx_feedback_message ON feedback(message_id);
CREATE INDEX idx_feedback_timestamp ON feedback(timestamp);

CREATE TABLE prompt_stats (
    id BIGSERIAL PRIMARY KEY,
    prompt_text VARCHAR(1000) NOT NULL UNIQUE,
    prompt_count BIGINT NOT NULL,
    last_used_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_prompt_stats_count_last_used ON prompt_stats(prompt_count, last_used_at);
