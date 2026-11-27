package httpapi

import (
	"encoding/json"
	"net/http"
	"net/url"

	"github.com/gin-gonic/gin"
	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

const HealthPath = "/health"

type Server struct {
	Service        *mechanic.Service
	Worker, Speech *url.URL
	Fleet          json.RawMessage
}

func (s *Server) Routes(dist string) http.Handler {
	router := gin.New()
	router.HandleMethodNotAllowed = true
	router.Use(gin.Recovery())
	read := []string{http.MethodGet, http.MethodHead}

	router.Match(read, HealthPath, health)
	docs := router.Group("/api/documents")
	document := documentProxy(s.Worker)
	docs.Match(read, "/:id", document)
	docs.Match(read, "/:id/pages/:page", document)

	session := router.Group("/", s.resolveSession)
	session.Match(read, "/api/vehicles", s.vehicles)
	session.POST("/agent/audio", s.audio)
	session.POST("/agent/run", s.run)
	session.POST("/agent/tickets/confirm", s.confirm)

	router.NoRoute(staticFiles(dist))
	return router
}
