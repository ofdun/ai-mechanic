package mechanic

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

type memoryTurns struct {
	Repository
	previous, saved *Conversation
	released        bool
}

func (m *memoryTurns) Begin(_ context.Context, _ Lease, _, _ string) (*Conversation, error) {
	return m.previous, nil
}

func (m *memoryTurns) Finish(_ Lease, c *Conversation) error {
	m.saved = c
	m.released = true
	return nil
}

type scriptedAgent struct {
	Agent
	err error
}

func (a scriptedAgent) Run(context.Context, Turn, string, *Conversation) (json.RawMessage, error) {
	return json.RawMessage(`{"message":"ответ","engine_model":"Д-01","diagnostic":null}`), a.err
}

func TestFollowUpKeepsDiagnosisAndLastFourMessages(t *testing.T) {
	repo := &memoryTurns{previous: &Conversation{Messages: []string{"1", "2", "3", "4"}, Diagnostic: json.RawMessage(`{"status":"anomaly"}`)}}
	s := Service{Repository: repo, Agent: scriptedAgent{}, Measurements: map[[2]string]bool{{"V", "S"}: true}}
	q := "почему"
	turn := Turn{Vehicle: "V", Session: "S", Message: &q}
	if _, err := s.Run(context.Background(), "mechanic", turn); err != nil {
		t.Fatal(err)
	}
	c := repo.saved
	if c.Diagnostic == nil || len(c.Messages) != 4 || c.Messages[0] != "3" || c.Messages[2] != q || c.Messages[3] != "ответ" {
		t.Fatalf("unexpected context: %+v", c)
	}
	repo.released = false
	s.Agent = scriptedAgent{err: errors.New("offline")}
	if _, err := s.Run(context.Background(), "mechanic", turn); err == nil || !repo.released || repo.saved != nil {
		t.Fatal("failed turn must release the conversation")
	}
	if _, err := s.Confirm(context.Background(), "mechanic", Decision{}); !errors.Is(err, ErrConfirmation) {
		t.Fatal(err)
	}
}

func TestInspectionOperationsLeaveTheConversationAlone(t *testing.T) {
	for _, command := range []string{"record_inspection", "get_diagnostic_history"} {
		repo := &memoryTurns{previous: &Conversation{Messages: []string{"question"}, Diagnostic: json.RawMessage(`{"status":"anomaly"}`)}}
		s := Service{Repository: repo, Agent: scriptedAgent{}, Measurements: map[[2]string]bool{{"V", "S"}: true}}
		turn := Turn{Vehicle: "V", Session: "S", Command: command}
		if command == "record_inspection" {
			turn.Inspection = json.RawMessage(`{"outcome":"confirmed"}`)
		}
		if _, err := s.Run(context.Background(), "mechanic", turn); err != nil || !repo.released || repo.saved != nil {
			t.Fatalf("%s altered the dialog or failed to release it: %v", command, err)
		}
	}
}
