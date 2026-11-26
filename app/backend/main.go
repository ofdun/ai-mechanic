package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ofdun/ai-mechanic/backend/internal/agent"
	"github.com/ofdun/ai-mechanic/backend/internal/httpapi"
	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
	"github.com/ofdun/ai-mechanic/backend/internal/sqlite"
)

func env(k, fallback string) string {
	if value := os.Getenv(k); value != "" {
		return value
	}
	return fallback
}

func main() {
	if len(os.Args) == 2 && os.Args[1] == "health" {
		c := http.Client{Timeout: 3 * time.Second}
		r, err := c.Get("http://127.0.0.1:8080/health")
		if err != nil {
			log.Fatal(err)
		}
		r.Body.Close()
		if r.StatusCode != 200 {
			os.Exit(1)
		}
		return
	}
	worker, err := url.Parse(env("AGENT_URL", "http://agent:8000"))
	if err != nil {
		log.Fatal(err)
	}
	db, err := sqlite.Open(env("ARTIFACTS_DIR", "/state"))
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()
	client := &agent.Client{URL: worker, HTTP: &http.Client{Timeout: 20 * time.Minute}}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	data, err := client.Fleet(ctx)
	if err != nil {
		log.Fatal(err)
	}
	measurements, err := mechanic.Measurements(data)
	if err != nil {
		log.Fatal(err)
	}
	s := httpapi.Server{Worker: worker, Fleet: data, Service: &mechanic.Service{Agent: client, Repository: db, Measurements: measurements}}
	httpServer := http.Server{Addr: env("LISTEN_ADDR", ":8080"), Handler: s.Routes(env("FRONTEND_DIST", "/app/frontend/dist")), ReadHeaderTimeout: 10 * time.Second}
	stopping, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	drained := make(chan struct{})
	go func() {
		<-stopping.Done()
		deadline, cancel := context.WithTimeout(context.Background(), 25*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(deadline); err != nil {
			log.Printf("shutdown: %v", err)
		}
		close(drained)
	}()
	if err := httpServer.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
	<-drained
}
