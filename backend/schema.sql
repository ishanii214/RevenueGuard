-- RevenueGuard PostgreSQL schema (Phase 6).
-- Applied idempotently by backend/db.apply_schema. Domain timestamps use
-- TIMESTAMP WITHOUT TIME ZONE so DB-loaded frames match the CSV frames'
-- naive datetime semantics exactly. investigation_results.investigated_at is
-- the only TIMESTAMPTZ column: it is operational metadata (when the API ran
-- the investigation) and never feeds back into any temporal logic.
--
-- investigation_results is a CURRENT-RESULT SNAPSHOT (one row per
-- transaction_id, upserted) — explicitly NOT a historical audit log.

CREATE TABLE IF NOT EXISTS customers (
    customer_id              TEXT PRIMARY KEY,
    signup_date              DATE NOT NULL,
    customer_segment         TEXT NOT NULL,
    country                  TEXT NOT NULL,
    preferred_payment_method TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id   TEXT PRIMARY KEY,
    customer_id      TEXT NOT NULL REFERENCES customers (customer_id),
    created_at       TIMESTAMP NOT NULL,
    amount           NUMERIC(12, 2) NOT NULL,
    currency         TEXT NOT NULL,
    payment_method   TEXT NOT NULL,
    status           TEXT NOT NULL,
    recovery_outcome TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions (customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions (status);

CREATE TABLE IF NOT EXISTS payment_attempts (
    attempt_id     TEXT PRIMARY KEY,
    transaction_id TEXT NOT NULL REFERENCES transactions (transaction_id),
    attempt_number INTEGER NOT NULL,
    attempted_at   TIMESTAMP NOT NULL,
    status         TEXT NOT NULL,
    payment_method TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_transaction ON payment_attempts (transaction_id);
CREATE INDEX IF NOT EXISTS idx_attempts_attempted_at ON payment_attempts (attempted_at);

CREATE TABLE IF NOT EXISTS payment_failures (
    failure_id              TEXT PRIMARY KEY,
    attempt_id              TEXT NOT NULL UNIQUE REFERENCES payment_attempts (attempt_id),
    transaction_id          TEXT NOT NULL REFERENCES transactions (transaction_id),
    customer_id             TEXT NOT NULL REFERENCES customers (customer_id),
    failed_at               TIMESTAMP NOT NULL,
    failure_reason          TEXT NOT NULL,
    processor_response_code TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_failures_transaction ON payment_failures (transaction_id);
CREATE INDEX IF NOT EXISTS idx_failures_attempted_at ON payment_failures (failed_at);

CREATE TABLE IF NOT EXISTS investigation_results (
    transaction_id        TEXT PRIMARY KEY REFERENCES transactions (transaction_id),
    prediction_time       TIMESTAMP NOT NULL,
    investigated_at       TIMESTAMPTZ NOT NULL,
    recommendation        TEXT NOT NULL,
    policy_decision       TEXT NOT NULL,
    final_action          TEXT NOT NULL,
    execution_authorized  BOOLEAN NOT NULL,
    policy_version        TEXT NOT NULL,
    result                JSONB NOT NULL
);
