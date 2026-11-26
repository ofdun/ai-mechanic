package sqlite

import (
	"context"
	"time"

	"gorm.io/gorm"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

func (s Store) Begin(ctx context.Context, l mechanic.Lease, vehicle, session string) (*mechanic.Conversation, error) {
	var row conversation
	err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		key := conversation{ActorID: l.Actor, Conversation: l.ID}
		err := tx.Where(&key).Attrs(conversation{Vehicle: vehicle, Session: session}).FirstOrCreate(&row).Error
		if err != nil {
			return err
		}
		if row.Vehicle != vehicle || row.Session != session {
			return mechanic.ErrMismatch
		}
		now := time.Now()
		if row.LeaseUntil > now.Unix() {
			return mechanic.ErrBusy
		}
		return tx.Model(&row).Select("Turn", "LeaseUntil").Updates(conversation{Turn: &l.Token, LeaseUntil: now.Add(mechanic.TurnLease).Unix()}).Error
	})
	if err != nil {
		return nil, err
	}
	return row.Context, nil
}

func (s Store) Finish(l mechanic.Lease, next *mechanic.Conversation) error {
	fields := []string{"Turn", "LeaseUntil"}
	if next != nil {
		fields = append(fields, "Context")
	}
	key := conversation{ActorID: l.Actor, Conversation: l.ID, Turn: &l.Token}
	update := s.db.Model(&conversation{}).Where(&key, "ActorID", "Conversation", "Turn").Select(fields).Updates(conversation{Context: next})
	if update.Error != nil {
		return update.Error
	}
	if update.RowsAffected != 1 {
		return mechanic.ErrMismatch
	}
	return nil
}
