package httpapi

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
)

func TestPublicFilesAndManualProxy(t *testing.T) {
	dist := t.TempDir()
	if err := os.WriteFile(filepath.Join(dist, "index.html"), []byte("mechanic"), 0600); err != nil {
		t.Fatal(err)
	}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Cookie") != "" || r.Header.Get("Authorization") != "" {
			t.Error("browser credentials reached the document worker")
		}
		if r.URL.RequestURI() != "/api/documents/manual/pages/2?view=image" {
			t.Errorf("unexpected target %s", r.URL.RequestURI())
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()
	target, _ := url.Parse(upstream.URL)
	s := Server{Worker: target}
	h := s.Routes(dist)
	for _, tc := range []struct {
		method, path string
		want         int
	}{
		{"GET", "/", 200}, {"GET", "/missing", 404}, {"HEAD", HealthPath, 200},
		{"GET", "/api/documents/manual/pages/2?view=image", 200}, {"POST", HealthPath, 405},
	} {
		r := httptest.NewRequest(tc.method, tc.path, nil)
		r.Header.Set("Cookie", "example=value")
		r.Header.Set("Authorization", "test")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != tc.want {
			t.Errorf("%s %s: got %d, want %d", tc.method, tc.path, w.Code, tc.want)
		}
	}
}
