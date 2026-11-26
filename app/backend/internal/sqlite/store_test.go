package sqlite

import (
	"context"
	"database/sql"
	_ "embed"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

//go:embed testdata/legacy.sql
var legacySchema string

func TestResumeExistingActorAndConversation(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	legacy, err := sql.Open("sqlite3", filepath.Join(dir, "web-sessions.sqlite3"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = legacy.Exec(legacySchema); err != nil {
		t.Fatal(err)
	}
	id := actorID("old-cookie")
	if _, err = legacy.Exec("INSERT INTO actors VALUES(?,?)", id, time.Now().Add(time.Hour).Unix()); err != nil {
		t.Fatal(err)
	}
	if _, err = legacy.Exec("INSERT INTO conversations(actor,conversation,vehicle,session,context) VALUES(?,?,?,?,?)", id, "chat", "car", "sample", `{"engine_model":"Д-01","recent_messages":["Вопрос"]}`); err != nil {
		t.Fatal(err)
	}
	legacy.Close()
	db, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	actor, token, err := db.Actor(ctx, "old-cookie")
	if actor != id || token != "" || err != nil {
		t.Fatalf("existing cookie lost: %v", err)
	}
	lease := mechanic.Lease{Actor: id, ID: "chat", Token: "first"}
	prev, err := db.Begin(ctx, lease, "car", "sample")
	if err != nil || prev.Engine != "Д-01" {
		t.Fatalf("missing saved context: %v", err)
	}
	if _, err = db.Begin(ctx, lease, "other", "sample"); !errors.Is(err, mechanic.ErrMismatch) {
		t.Fatal(err)
	}
	if _, err = db.Begin(ctx, lease, "car", "sample"); !errors.Is(err, mechanic.ErrBusy) {
		t.Fatal(err)
	}
	wrong := lease
	wrong.Token = "other"
	if err = db.Finish(wrong, nil); !errors.Is(err, mechanic.ErrMismatch) {
		t.Fatal(err)
	}
	if err = db.Finish(lease, nil); err != nil {
		t.Fatal(err)
	}
	prev, err = db.Begin(ctx, lease, "car", "sample")
	if err != nil || prev.Messages[0] != "Вопрос" {
		t.Fatalf("failed turn changed context: %v", err)
	}
}

func TestConcurrentTurnsAndExpiredActors(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	id, token, err := store.Actor(ctx, "")
	if err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	results := make(chan error, 2)
	for _, owner := range []string{"first", "second"} {
		go func(owner string) {
			<-start
			_, err := store.Begin(ctx, mechanic.Lease{Actor: id, ID: "chat", Token: owner}, "car", "sample")
			results <- err
		}(owner)
	}
	close(start)
	first, second := <-results, <-results
	if !((first == nil && errors.Is(second, mechanic.ErrBusy)) || (second == nil && errors.Is(first, mechanic.ErrBusy))) {
		t.Fatalf("turn results: %v, %v", first, second)
	}
	if err := store.db.Model(&actor{Actor: id}).Update("expires", 0).Error; err != nil {
		t.Fatal(err)
	}
	next, _, err := store.Actor(ctx, token)
	if err != nil || next == id {
		t.Fatalf("expired actor reused: %v", err)
	}
	var count int64
	if err := store.db.Model(&conversation{}).Where(&conversation{ActorID: id}).Count(&count).Error; err != nil || count != 0 {
		t.Fatalf("expired actor has %d conversations: %v", count, err)
	}
}
