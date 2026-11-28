package httpapi

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestPlatePhotoIsForwardedToWorkerOnly(t *testing.T) {
	photo := []byte("\xff\xd8jpeg")
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if r.URL.Path != "/plate" || r.Header.Get("Content-Type") != "image/jpeg" || !bytes.Equal(body, photo) {
			t.Errorf("unexpected worker request %s %q", r.URL.Path, r.Header.Get("Content-Type"))
		}
		if r.Header.Get("Cookie") != "" {
			t.Error("browser cookie reached the worker")
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"readings":["M333YK47"],"candidates":[],"reliable":false}`))
	}))
	defer upstream.Close()
	target, _ := url.Parse(upstream.URL)
	s := Server{Worker: target}
	router := gin.New()
	router.POST("/api/plate", s.plate)

	for _, tc := range []struct {
		name, contentType string
		body              []byte
		want              int
	}{
		{"photo", "image/jpeg", photo, http.StatusOK},
		{"not an image", "text/plain", photo, http.StatusUnsupportedMediaType},
		{"empty", "image/jpeg", nil, http.StatusBadRequest},
		{"too large", "image/jpeg", make([]byte, maxPlatePhoto+1), http.StatusRequestEntityTooLarge},
	} {
		req := httptest.NewRequest(http.MethodPost, "/api/plate", bytes.NewReader(tc.body))
		req.Header.Set("Content-Type", tc.contentType)
		req.Header.Set("Cookie", "mechanic=session")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != tc.want {
			t.Errorf("%s: got %d, want %d", tc.name, rec.Code, tc.want)
		}
	}
}
