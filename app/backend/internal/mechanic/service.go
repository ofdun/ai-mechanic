package mechanic

import (
	"context"
	"encoding/json"
	"log"
)

func (s *Service) Run(ctx context.Context, actor string, t Turn) (json.RawMessage, error) {
	if err := t.normalize(); err != nil {
		return nil, err
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
	next, err := nextContext(actor, t, prev, data)
	if err != nil {
		return nil, err
	}
	if err = s.Repository.Finish(l, next); err != nil {
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
