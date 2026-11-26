package sqlite

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

func TestResumeExistingActorAndConversation(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	db, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	id := actorID("old-cookie")
	if _, err = db.Exec("INSERT INTO actors VALUES(?,?)", id, time.Now().Add(time.Hour).Unix()); err != nil {
		t.Fatal(err)
	}
	if _, err = db.Exec("INSERT INTO conversations(actor,conversation,vehicle,session,context) VALUES(?,?,?,?,?)", id, "chat", "car", "sample", `{"engine_model":"Д-01","recent_messages":["Вопрос"]}`); err != nil {
		t.Fatal(err)
	}
	db.Close()
	db, err = Open(dir)
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
