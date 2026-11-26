package sqlite

import (
	"context"
	"crypto/sha256"
	"fmt"
	"time"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

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
