-- Auto-run by the postgres container on first start.
CREATE SEQUENCE IF NOT EXISTS ticket_seq START 1000;

CREATE TABLE IF NOT EXISTS users (
    telegram_user_id BIGINT PRIMARY KEY,
    name             TEXT,
    phone            TEXT,
    language         TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS units (
    id          SERIAL PRIMARY KEY,
    compound    TEXT NOT NULL,
    building    TEXT NOT NULL,
    unit_number TEXT NOT NULL,
    UNIQUE (compound, building, unit_number)
);

CREATE TABLE IF NOT EXISTS user_units (
    telegram_user_id BIGINT REFERENCES users ON DELETE CASCADE,
    unit_id          INT REFERENCES units,
    confirmed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (telegram_user_id, unit_id)
);

CREATE TABLE IF NOT EXISTS service_requests (
    id                SERIAL PRIMARY KEY,
    ticket_number     TEXT UNIQUE NOT NULL DEFAULT ('SR-' || nextval('ticket_seq')),
    telegram_user_id  BIGINT REFERENCES users ON DELETE SET NULL,
    unit_id           INT REFERENCES units,
    issue_description TEXT NOT NULL,
    category          TEXT NOT NULL,
    urgency           TEXT NOT NULL,
    confidence        REAL,
    needs_review      BOOLEAN NOT NULL DEFAULT FALSE,
    status            TEXT NOT NULL DEFAULT 'open',   -- open | in_progress | closed
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at         TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id         BIGSERIAL PRIMARY KEY,
    sr_id      INT REFERENCES service_requests ON DELETE CASCADE,
    chat_id    BIGINT NOT NULL,
    role       TEXT NOT NULL,          -- user | assistant
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id         BIGSERIAL PRIMARY KEY,
    sr_id      INT REFERENCES service_requests ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    category   TEXT,
    urgency    TEXT,
    confidence REAL,
    latency_ms INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sr_created ON service_requests (created_at);
CREATE INDEX IF NOT EXISTS idx_sr_status ON service_requests (status);
