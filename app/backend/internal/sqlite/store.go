package sqlite

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

type Store struct{ *sql.DB }

func Open(dir string) (Store, error) {
	if err := os.MkdirAll(dir, 0700); err != nil {
		return Store{}, err
	}
	path, err := filepath.Abs(filepath.Join(dir, "web-sessions.sqlite3"))
	if err != nil {
		return Store{}, err
	}
	dsn := url.URL{Scheme: "file", Path: path, RawQuery: "_busy_timeout=10000&_foreign_keys=on&_txlock=immediate"}
	db, err := sql.Open("sqlite3", dsn.String())
	if err != nil {
		return Store{}, err
	}
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS actors(actor TEXT PRIMARY KEY,expires INTEGER NOT NULL);
 CREATE TABLE IF NOT EXISTS conversations(actor TEXT NOT NULL REFERENCES actors ON DELETE CASCADE,
 conversation TEXT NOT NULL,vehicle TEXT NOT NULL,session TEXT NOT NULL,context TEXT,turn TEXT,
 lease_until INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(actor,conversation));`)
	if err != nil {
		db.Close()
		return Store{}, err
	}
	return Store{db}, nil
}

func actorID(token string) string { return fmt.Sprintf("%x", sha256.Sum256([]byte(token))) }

func (s Store) Actor(ctx context.Context, token string) (id, newToken string, err error) {
	now := time.Now()
	if token != "" {
		id = actorID(token)
		var valid bool
		err = s.QueryRowContext(ctx, "SELECT EXISTS(SELECT 1 FROM actors WHERE actor=? AND expires>?)", id, now.Unix()).Scan(&valid)
		if err != nil || valid {
			return id, "", err
		}
	}
	newToken = mechanic.RandomID(32)
	id = actorID(newToken)
	tx, err := s.BeginTx(ctx, nil)
	if err != nil {
		return "", "", err
	}
	defer tx.Rollback()
	if _, err = tx.ExecContext(ctx, "DELETE FROM actors WHERE expires<=?", now.Unix()); err != nil {
		return "", "", err
	}
	if _, err = tx.ExecContext(ctx, "INSERT INTO actors VALUES (?,?)", id, now.Add(mechanic.SessionLifetime).Unix()); err != nil {
		return "", "", err
	}
	return id, newToken, tx.Commit()
}

func (s Store) Begin(ctx context.Context, l mechanic.Lease, vehicle, session string) (*mechanic.Conversation, error) {
	tx, err := s.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	_, err = tx.ExecContext(ctx, "INSERT OR IGNORE INTO conversations(actor,conversation,vehicle,session) VALUES (?,?,?,?)", l.Actor, l.ID, vehicle, session)
	if err != nil {
		return nil, err
	}
	var v, m string
	var saved sql.NullString
	var until int64
	err = tx.QueryRowContext(ctx, "SELECT vehicle,session,context,lease_until FROM conversations WHERE actor=? AND conversation=?", l.Actor, l.ID).Scan(&v, &m, &saved, &until)
	if err != nil {
		return nil, err
	}
	if v != vehicle || m != session {
		return nil, mechanic.ErrMismatch
	}
	now := time.Now()
	if until > now.Unix() {
		return nil, mechanic.ErrBusy
	}
	var prev *mechanic.Conversation
	if saved.Valid {
		if err = json.Unmarshal([]byte(saved.String), &prev); err != nil {
			return nil, err
		}
	}
	_, err = tx.ExecContext(ctx, "UPDATE conversations SET turn=?,lease_until=? WHERE actor=? AND conversation=?", l.Token, now.Add(30*time.Minute).Unix(), l.Actor, l.ID)
	if err != nil {
		return nil, err
	}
	return prev, tx.Commit()
}

func (s Store) Finish(l mechanic.Lease, next *mechanic.Conversation) error {
	var data any
	if next != nil {
		b, err := json.Marshal(next)
		if err != nil {
			return err
		}
		data = string(b)
	}
	result, err := s.Exec("UPDATE conversations SET context=COALESCE(?,context),turn=NULL,lease_until=0 WHERE actor=? AND conversation=? AND turn=?", data, l.Actor, l.ID, l.Token)
	if err != nil {
		return err
	}
	n, err := result.RowsAffected()
	if err == nil && n != 1 {
		return mechanic.ErrMismatch
	}
	return err
}
