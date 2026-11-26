package httpapi

import (
	"encoding/json"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

const HealthPath = "/health"

type Server struct {
	Service *mechanic.Service
	Worker  *url.URL
	Fleet   json.RawMessage
}

func (s *Server) Routes(dist string) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET "+HealthPath, func(w http.ResponseWriter, r *http.Request) { reply(w, 200, []byte(`{"status":"ok"}`)) })
	mux.HandleFunc("GET /api/vehicles", s.session(func(w http.ResponseWriter, r *http.Request, _ string) { reply(w, 200, s.Fleet) }))
	mux.HandleFunc("POST /agent/run", s.session(s.run))
	mux.HandleFunc("POST /agent/tickets/confirm", s.session(s.confirm))
	docs := &httputil.ReverseProxy{Rewrite: func(p *httputil.ProxyRequest) {
		p.SetURL(s.Worker)
		p.Out.Header.Del("Cookie")
		p.Out.Header.Del("Authorization")
	}}
	mux.Handle("GET /api/documents/{id}", docs)
	mux.Handle("GET /api/documents/{id}/pages/{page}", docs)
	assets := http.FileServer(http.Dir(dist))
	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		path := filepath.Join(dist, filepath.FromSlash(r.URL.Path))
		fi, err := os.Stat(path)
		if err != nil || (fi.IsDir() && r.URL.Path != "/") {
			http.NotFound(w, r)
			return
		}
		assets.ServeHTTP(w, r)
	})
	return mux
}
