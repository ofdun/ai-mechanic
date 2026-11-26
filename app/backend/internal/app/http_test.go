package app

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProbeUsesListeningPort(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer s.Close()
	if err := probe(context.Background(), s.Listener.Addr().String()); err != nil {
		t.Fatal(err)
	}
}

func TestInvalidAgentSettings(t *testing.T) {
	for _, tc := range []struct{ url, timeout string }{
		{"/agent", "20m"}, {"ftp://agent", "20m"}, {"http://agent", "0s"}, {"http://agent", "bad"}, {"http://agent", "30m"},
	} {
		t.Run(fmt.Sprintf("%s_%s", tc.url, tc.timeout), func(t *testing.T) {
			t.Setenv("AGENT_URL", tc.url)
			t.Setenv("WEB_AGENT_TIMEOUT", tc.timeout)
			if _, err := loadConfig(); err == nil {
				t.Fatal("expected configuration error")
			}
		})
	}
}
