package httpapi

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

func reply(w http.ResponseWriter, status int, data []byte) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(data)
}

const maxBodyBytes = 1 << 20

func decode(w http.ResponseWriter, r *http.Request, dest any) bool {
	d := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	d.DisallowUnknownFields()
	if err := d.Decode(dest); err != nil || d.Decode(new(any)) != io.EOF {
		problem(w, mechanic.ErrInput)
		return false
	}
	return true
}
