package sqlite

import (
	"context"
	"crypto/sha256"
	"fmt"
	"time"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

func actorID(token string) string { return fmt.Sprintf("%x", sha256.Sum256([]byte(token))) }

func (s Store) Actor(ctx context.Context, token string) (string, string, error) {
	db := s.db.WithContext(ctx)
	now := time.Now()
	if token != "" {
		id := actorID(token)
		var count int64
		err := db.Model(&actor{}).Where(&actor{Actor: id}).Where(clause.Gt{Column: "expires", Value: now.Unix()}).Count(&count).Error
		if err != nil {
			return "", "", err
		}
		if count > 0 {
			return id, "", nil
		}
	}
	token = mechanic.RandomID(32)
	record := actor{Actor: actorID(token), Expires: now.Add(mechanic.SessionLifetime).Unix()}
	err := db.Transaction(func(tx *gorm.DB) error {
		if err := tx.Where(clause.Lte{Column: "expires", Value: now.Unix()}).Delete(&actor{}).Error; err != nil {
			return err
		}
		return tx.Create(&record).Error
	})
	if err != nil {
		return "", "", err
	}
	return record.Actor, token, nil
}
