package mechanic

import (
	"encoding/json"
)

const (
	maxExcerptRunes = 1500
	historySize     = 4
)

func excerpt(s string) string {
	r := []rune(s)
	if len(r) > maxExcerptRunes {
		r = r[:maxExcerptRunes]
	}
	return string(r)
}

func nextContext(actor string, t Turn, prev *Conversation, data json.RawMessage) (*Conversation, error) {
	if t.Command == "record_inspection" || t.Command == "get_diagnostic_history" {
		return nil, nil // Explicit form and history operations do not enter the chat transcript.
	}
	var resp struct {
		Message    string
		Engine     string `json:"engine_model"`
		Diagnostic json.RawMessage
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	next := Conversation{Actor: actor, ID: t.ID, Vehicle: t.Vehicle, Session: t.Session, Engine: resp.Engine, Diagnostic: resp.Diagnostic, Messages: []string{}}
	if prev != nil {
		next.Messages = prev.Messages
		if string(next.Diagnostic) == "null" {
			next.Diagnostic = prev.Diagnostic
		}
	}
	if t.Message != nil {
		next.Messages = append(next.Messages, excerpt(*t.Message))
	}
	next.Messages = append(next.Messages, excerpt(resp.Message))
	if len(next.Messages) > historySize {
		next.Messages = next.Messages[len(next.Messages)-historySize:]
	}
	return &next, nil
}
