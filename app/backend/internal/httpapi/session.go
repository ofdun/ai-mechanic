package httpapi

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

const sessionCookie = "mechanic_session"
const actorKey = "session.actor"

func (s *Server) resolveSession(c *gin.Context) {
	token, _ := c.Cookie(sessionCookie)
	id, fresh, err := s.Service.ResolveSession(c.Request.Context(), token)
	if err != nil {
		c.Abort()
		problem(c, err)
		return
	}
	if fresh != "" {
		http.SetCookie(c.Writer, &http.Cookie{Name: sessionCookie, Value: fresh, Path: "/", HttpOnly: true, Secure: c.Request.TLS != nil, SameSite: http.SameSiteStrictMode, MaxAge: int(mechanic.SessionLifetime.Seconds())})
	}
	c.Header("Cache-Control", "private, no-store")
	c.Set(actorKey, id)
}
