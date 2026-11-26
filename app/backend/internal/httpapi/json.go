package httpapi

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

const jsonType = "application/json"

const maxBodyBytes = 1 << 20

func decode(c *gin.Context, dest any) bool {
	d := json.NewDecoder(http.MaxBytesReader(c.Writer, c.Request.Body, maxBodyBytes))
	d.DisallowUnknownFields()
	if err := d.Decode(dest); err != nil || d.Decode(new(any)) != io.EOF {
		problem(c, mechanic.ErrInput)
		return false
	}
	return true
}
