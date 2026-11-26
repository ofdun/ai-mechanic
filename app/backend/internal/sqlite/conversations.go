package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"time"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

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
	_, err = tx.ExecContext(ctx, "UPDATE conversations SET turn=?,lease_until=? WHERE actor=? AND conversation=?", l.Token, now.Add(mechanic.TurnLease).Unix(), l.Actor, l.ID)
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
