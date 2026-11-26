package sqlite

import (
	"database/sql"
	_ "embed"
	_ "github.com/mattn/go-sqlite3"
	"net/url"
	"os"
	"path/filepath"
)

//go:embed schema.sql
var schema string

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
	_, err = db.Exec(schema)
	if err != nil {
		db.Close()
		return Store{}, err
	}
	return Store{db}, nil
}
