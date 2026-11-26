package app

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"time"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

type config struct {
	listenAddr, frontendDir, stateDir string
	agentURL                          *url.URL
	agentTimeout                      time.Duration
}

func setting(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func loadConfig() (config, error) {
	c := config{listenAddr: setting("LISTEN_ADDR", ":8080"), frontendDir: setting("FRONTEND_DIST", "/app/frontend/dist"), stateDir: setting("ARTIFACTS_DIR", "/state")}
	u, err := url.Parse(setting("AGENT_URL", "http://agent:8000"))
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Hostname() == "" || u.RawQuery != "" || u.Fragment != "" {
		return c, fmt.Errorf("AGENT_URL must be an HTTP or HTTPS base URL")
	}
	c.agentURL = u
	c.agentTimeout, err = time.ParseDuration(setting("WEB_AGENT_TIMEOUT", "20m"))
	if err != nil || c.agentTimeout <= 0 || c.agentTimeout >= mechanic.TurnLease {
		return c, fmt.Errorf("WEB_AGENT_TIMEOUT must be positive and shorter than %s", mechanic.TurnLease)
	}
	_, _, err = net.SplitHostPort(c.listenAddr)
	if err != nil {
		return c, fmt.Errorf("LISTEN_ADDR: %w", err)
	}
	return c, nil
}
