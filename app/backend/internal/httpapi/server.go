package httpapi

import (
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"

	"github.com/ofdun/ai-mechanic/backend/internal/agent"
	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

type Server struct {
	Service *mechanic.Service
	Worker  *url.URL
	Fleet   json.RawMessage
}

func reply(w http.ResponseWriter, status int, data []byte) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(data)
}

func fail(w http.ResponseWriter, status int, code, msg string) {
	b, _ := json.Marshal(map[string]any{"detail": map[string]any{"code": code, "message": msg, "retryable": code == "CONVERSATION_BUSY" || status == 503}})
	reply(w, status, b)
}

type actorHandler func(http.ResponseWriter, *http.Request, string)

func (s *Server) session(next actorHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var token string
		if cookie, err := r.Cookie("mechanic_session"); err == nil {
			token = cookie.Value
		}
		id, fresh, err := s.Service.ResolveSession(r.Context(), token)
		if err != nil {
			s.problem(w, err)
			return
		}
		if fresh != "" {
			http.SetCookie(w, &http.Cookie{Name: "mechanic_session", Value: fresh, Path: "/", HttpOnly: true, Secure: r.TLS != nil, SameSite: http.SameSiteStrictMode, MaxAge: int(mechanic.SessionLifetime.Seconds())})
		}
		w.Header().Set("Cache-Control", "private, no-store")
		next(w, r, id)
	}
}

func (s *Server) problem(w http.ResponseWriter, err error) {
	var remote *agent.ResponseError
	switch {
	case errors.As(err, &remote):
		reply(w, remote.Status, remote.Body)
	case errors.Is(err, mechanic.ErrConfirmation):
		fail(w, 422, "CONFIRMATION_REQUIRED", "Нужно явное подтверждение механика.")
	case errors.Is(err, mechanic.ErrInput):
		fail(w, 422, "INVALID_INPUT", "Проверьте параметры запроса.")
	case errors.Is(err, mechanic.ErrMeasurement):
		fail(w, 404, "VEHICLE_OR_SESSION_NOT_FOUND", "Замер не найден для выбранной машины.")
	case errors.Is(err, mechanic.ErrBusy):
		fail(w, 409, "CONVERSATION_BUSY", "В этом диалоге ещё готовится ответ.")
	case errors.Is(err, mechanic.ErrMismatch):
		fail(w, 409, "CONTEXT_MISMATCH", "Для выбранной машины или проверки нужен новый диалог.")
	default:
		log.Printf("request failed: %v", err)
		fail(w, 503, "SERVICE_UNAVAILABLE", "Сервис временно недоступен.")
	}
}

func decode(w http.ResponseWriter, r *http.Request, dest any) bool {
	d := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	d.DisallowUnknownFields()
	if err := d.Decode(dest); err != nil || d.Decode(new(any)) != io.EOF {
		fail(w, 422, "INVALID_INPUT", "Проверьте параметры запроса.")
		return false
	}
	return true
}

func (s *Server) run(w http.ResponseWriter, r *http.Request, actor string) {
	var t mechanic.Turn
	if !decode(w, r, &t) {
		return
	}
	data, err := s.Service.Run(r.Context(), actor, t)
	if err != nil {
		s.problem(w, err)
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
		s.problem(w, err)
		return
	}
	reply(w, 200, data)
}

func (s *Server) Routes(dist string) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { reply(w, 200, []byte(`{"status":"ok"}`)) })
	mux.HandleFunc("GET /api/vehicles", s.session(func(w http.ResponseWriter, r *http.Request, _ string) { reply(w, 200, s.Fleet) }))
	mux.HandleFunc("POST /agent/run", s.session(s.run))
	mux.HandleFunc("POST /agent/tickets/confirm", s.session(s.confirm))
	docs := &httputil.ReverseProxy{Rewrite: func(p *httputil.ProxyRequest) {
		p.SetURL(s.Worker)
		p.Out.Header.Del("Cookie")
		p.Out.Header.Del("Authorization")
	}}
	mux.Handle("GET /api/documents/{id}", docs)
	mux.Handle("GET /api/documents/{id}/pages/{page}", docs)
	assets := http.FileServer(http.Dir(dist))
	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		path := filepath.Join(dist, filepath.FromSlash(r.URL.Path))
		fi, err := os.Stat(path)
		if err != nil || (fi.IsDir() && r.URL.Path != "/") {
			http.NotFound(w, r)
			return
		}
		assets.ServeHTTP(w, r)
	})
	return mux
}
