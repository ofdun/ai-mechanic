package httpapi

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

func (s *Server) run(c *gin.Context) {
	var t mechanic.Turn
	if !decode(c, &t) {
		return
	}
	data, err := s.Service.Run(c.Request.Context(), c.GetString(actorKey), t)
	if err != nil {
		problem(c, err)
		return
	}
	c.Data(http.StatusOK, jsonType, data)
}

func (s *Server) confirm(c *gin.Context) {
	var payload mechanic.Decision
	if !decode(c, &payload) {
		return
	}
	data, err := s.Service.Confirm(c.Request.Context(), c.GetString(actorKey), payload)
	if err != nil {
		problem(c, err)
		return
	}
	c.Data(http.StatusOK, jsonType, data)
}

func health(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"status": "ok"}) }

func (s *Server) vehicles(c *gin.Context) { c.Data(http.StatusOK, jsonType, s.Fleet) }
