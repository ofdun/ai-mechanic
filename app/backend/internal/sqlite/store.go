package sqlite

import (
	"database/sql"
	"net/url"
	"os"
	"path/filepath"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

type actor struct {
	Actor   string `gorm:"primaryKey"`
	Expires int64  `gorm:"not null"`
}

type conversation struct {
	ActorID      string                 `gorm:"column:actor;primaryKey;not null"`
	Conversation string                 `gorm:"primaryKey;not null"`
	Owner        actor                  `gorm:"foreignKey:ActorID;references:Actor;constraint:OnDelete:CASCADE"`
	Vehicle      string                 `gorm:"not null"`
	Session      string                 `gorm:"not null"`
	Context      *mechanic.Conversation `gorm:"serializer:json;type:text"`
	Turn         *string
	LeaseUntil   int64 `gorm:"not null;default:0"`
}

type Store struct {
	db   *gorm.DB
	pool *sql.DB
}

func Open(dir string) (Store, error) {
	if err := os.MkdirAll(dir, 0700); err != nil {
		return Store{}, err
	}
	path, err := filepath.Abs(filepath.Join(dir, "web-sessions.sqlite3"))
	if err != nil {
		return Store{}, err
	}
	dsn := url.URL{Scheme: "file", Path: path, RawQuery: "_busy_timeout=10000&_foreign_keys=on&_txlock=immediate"}
	db, err := gorm.Open(sqlite.Open(dsn.String()), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		return Store{}, err
	}
	pool, err := db.DB()
	if err != nil {
		return Store{}, err
	}
	// Existing deployments share this schema with older releases during rollback.
	err = db.Transaction(func(tx *gorm.DB) error {
		m := tx.Migrator()
		for _, model := range []any{&actor{}, &conversation{}} {
			if !m.HasTable(model) {
				if err := m.CreateTable(model); err != nil {
					return err
				}
			}
		}
		return nil
	})
	if err != nil {
		pool.Close()
		return Store{}, err
	}
	return Store{db: db, pool: pool}, nil
}

func (s Store) Close() error { return s.pool.Close() }
