package app

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/ofdun/ai-mechanic/backend/internal/agent"
	"github.com/ofdun/ai-mechanic/backend/internal/httpapi"
	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
	"github.com/ofdun/ai-mechanic/backend/internal/sqlite"
)

const fleetLoadTimeout = 10 * time.Second

func Run(ctx context.Context, args []string) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	if len(args) == 1 && args[0] == "health" {
		return probe(ctx, cfg.listenAddr)
	}
	if len(args) > 0 {
		return fmt.Errorf("usage: mechanic [health]")
	}
	store, err := sqlite.Open(cfg.stateDir)
	if err != nil {
		return fmt.Errorf("open web sessions: %w", err)
	}
	defer store.Close()
	client := &agent.Client{URL: cfg.agentURL, HTTP: &http.Client{Timeout: cfg.agentTimeout}}
	loadCtx, cancel := context.WithTimeout(ctx, fleetLoadTimeout)
	fleet, err := client.Fleet(loadCtx)
	cancel()
	if err != nil {
		return fmt.Errorf("load fleet: %w", err)
	}
	measurements, err := mechanic.Measurements(fleet)
	if err != nil {
		return fmt.Errorf("decode fleet: %w", err)
	}
	service := &mechanic.Service{Repository: store, Agent: client, Measurements: measurements}
	api := httpapi.Server{Worker: cfg.agentURL, Fleet: fleet, Service: service}
	return serve(ctx, cfg.listenAddr, api.Routes(cfg.frontendDir))
}
