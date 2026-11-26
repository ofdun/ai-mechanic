package httpapi

import (
	"net/http"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

func (s *Server) run(w http.ResponseWriter, r *http.Request, actor string) {
	var t mechanic.Turn
	if !decode(w, r, &t) {
		return
	}
	data, err := s.Service.Run(r.Context(), actor, t)
	if err != nil {
		problem(w, err)
		return
	}
	reply(w, 200, data)
}

func (s *Server) confirm(w http.ResponseWriter, r *http.Request, actor string) {
	var payload mechanic.Decision
	if !decode(w, r, &payload) {
		return
	}
	data, err := s.Service.Confirm(r.Context(), actor, payload)
	if err != nil {
		problem(w, err)
		return
	}
	reply(w, 200, data)
}
