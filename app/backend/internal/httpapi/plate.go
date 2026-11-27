package httpapi

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

const maxPlatePhoto = 8 << 20

// plate forwards a registration plate photo to the agent worker for local OCR.
// The photo is neither stored nor logged; the reply only suggests vehicles.
func (s *Server) plate(c *gin.Context) {
	if !strings.HasPrefix(c.ContentType(), "image/") {
		c.Status(http.StatusUnsupportedMediaType)
		return
	}
	photo, err := io.ReadAll(http.MaxBytesReader(c.Writer, c.Request.Body, maxPlatePhoto))
	if err != nil {
		var limit *http.MaxBytesError
		if errors.As(err, &limit) {
			c.Status(http.StatusRequestEntityTooLarge)
		} else {
			c.Status(http.StatusBadRequest)
		}
		return
	}
	if len(photo) == 0 {
		c.Status(http.StatusBadRequest)
		return
	}
	ctx, cancel := context.WithTimeout(c.Request.Context(), 45*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.Worker.JoinPath("plate").String(), bytes.NewReader(photo))
	if err != nil {
		c.Status(http.StatusBadGateway)
		return
	}
	req.Header.Set("Content-Type", c.ContentType())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fail(c, http.StatusBadGateway, "PLATE_OCR_FAILED", "Номер не распознан. Выберите машину из списка.", true)
		return
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if err != nil {
		c.Status(http.StatusBadGateway)
		return
	}
	c.Data(resp.StatusCode, jsonType, data)
}
