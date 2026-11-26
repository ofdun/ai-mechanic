package httpapi

import (
	"net/http"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

const sessionCookie = "mechanic_session"

type actorHandler func(http.ResponseWriter, *http.Request, string)

func (s *Server) session(next actorHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var token string
		if cookie, err := r.Cookie(sessionCookie); err == nil {
			token = cookie.Value
		}
		id, fresh, err := s.Service.ResolveSession(r.Context(), token)
		if err != nil {
			problem(w, err)
			return
		}
		if fresh != "" {
			http.SetCookie(w, &http.Cookie{Name: sessionCookie, Value: fresh, Path: "/", HttpOnly: true, Secure: r.TLS != nil, SameSite: http.SameSiteStrictMode, MaxAge: int(mechanic.SessionLifetime.Seconds())})
		}
		w.Header().Set("Cache-Control", "private, no-store")
		next(w, r, id)
	}
}
