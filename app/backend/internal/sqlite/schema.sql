CREATE TABLE IF NOT EXISTS actors (
    actor TEXT PRIMARY KEY,
    expires INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    actor TEXT NOT NULL REFERENCES actors ON DELETE CASCADE,
    conversation TEXT NOT NULL,
    vehicle TEXT NOT NULL,
    session TEXT NOT NULL,
    context TEXT,
    turn TEXT,
    lease_until INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (actor, conversation)
);
