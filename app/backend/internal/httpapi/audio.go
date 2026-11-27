package httpapi

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

func (s *Server) audio(c *gin.Context) {
	const maxRecording = 2 << 20
	recording, err := io.ReadAll(http.MaxBytesReader(c.Writer, c.Request.Body, maxRecording))
	if err != nil {
		var limit *http.MaxBytesError
		if errors.As(err, &limit) {
			c.Status(http.StatusRequestEntityTooLarge)
		} else {
			c.Status(http.StatusBadRequest)
		}
		return
	}
	if len(recording) == 0 {
		c.Status(http.StatusBadRequest)
		return
	}
	target := s.Speech.JoinPath("agent/audio")
	query := target.Query()
	query.Set("language", c.DefaultQuery("language", "ru"))
	target.RawQuery = query.Encode()
	ctx, cancel := context.WithTimeout(c.Request.Context(), 60*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target.String(), bytes.NewReader(recording))
	if err != nil {
		c.Status(http.StatusBadGateway)
		return
	}
	req.Header.Set("Content-Type", c.ContentType())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		c.Status(http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		c.Status(resp.StatusCode)
		return
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if err != nil {
		c.Status(http.StatusBadGateway)
		return
	}
	c.Data(http.StatusOK, jsonType, data)
}
