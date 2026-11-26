package mechanic

import (
	"context"
	"encoding/json"
	"log"
	"strings"
	"unicode/utf8"
)

func excerpt(s string) string {
	r := []rune(s)
	if len(r) > 1500 {
		r = r[:1500]
	}
	return string(r)
}

func (s *Service) Run(ctx context.Context, actor string, t Turn) (json.RawMessage, error) {
	t.Vehicle = strings.TrimSpace(t.Vehicle)
	t.Session = strings.TrimSpace(t.Session)
	t.ID = strings.TrimSpace(t.ID)
	if t.Message != nil {
		clean := strings.TrimSpace(*t.Message)
		if clean == "" || utf8.RuneCountInString(clean) > 8000 {
			return nil, ErrInput
		}
		t.Message = &clean
	}
	switch t.Command {
	case "", "auto":
		if t.Message == nil {
			return nil, ErrInput
		}
	case "diagnose", "draft_ticket":
	default:
		return nil, ErrInput
	}
	if t.Vehicle == "" || t.Session == "" {
		return nil, ErrInput
	}
	if !s.Measurements[[2]string{t.Vehicle, t.Session}] {
		return nil, ErrMeasurement
	}
	if t.ID == "" {
		t.ID = RandomID(16)
	}
	l := Lease{actor, t.ID, RandomID(16)}
	prev, err := s.Repository.Begin(ctx, l, t.Vehicle, t.Session)
	if err != nil {
		return nil, err
	}
	saved := false
	defer func() {
		if !saved {
			if err := s.Repository.Finish(l, nil); err != nil {
				log.Printf("release turn: %v", err)
			}
		}
	}()
	data, err := s.Agent.Run(ctx, t, actor, prev)
	if err != nil {
		return nil, err
	}
	var resp struct {
		Message    string
		Engine     string `json:"engine_model"`
		Diagnostic json.RawMessage
	}
	if err = json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	next := Conversation{Actor: actor, ID: t.ID, Vehicle: t.Vehicle, Session: t.Session, Engine: resp.Engine, Diagnostic: resp.Diagnostic, Messages: []string{}}
	if prev != nil {
		next.Messages = prev.Messages
		if string(next.Diagnostic) == "null" {
			next.Diagnostic = prev.Diagnostic
		}
	}
	if t.Message != nil && strings.TrimSpace(*t.Message) != "" {
		next.Messages = append(next.Messages, excerpt(strings.TrimSpace(*t.Message)))
	}
	next.Messages = append(next.Messages, excerpt(resp.Message))
	if len(next.Messages) > 4 {
		next.Messages = next.Messages[len(next.Messages)-4:]
	}
	if err = s.Repository.Finish(l, &next); err != nil {
		return nil, err
	}
	saved = true
	return data, nil
}

func (s *Service) ResolveSession(ctx context.Context, token string) (string, string, error) {
	return s.Repository.Actor(ctx, token)
}

func (s *Service) Confirm(ctx context.Context, actor string, decision Decision) (json.RawMessage, error) {
	if decision.Confirmed == nil || !*decision.Confirmed {
		return nil, ErrConfirmation
	}
	return s.Agent.Confirm(ctx, decision, actor)
}
