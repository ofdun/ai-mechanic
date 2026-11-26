package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

type Client struct {
	URL  *url.URL
	HTTP *http.Client
}

type ResponseError struct {
	Status int
	Body   []byte
}

func (e *ResponseError) Error() string { return fmt.Sprintf("agent returned %d", e.Status) }

func (s *Client) call(ctx context.Context, path string, body any) (json.RawMessage, error) {
	method := http.MethodGet
	var reader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(b)
		method = http.MethodPost
	}
	req, err := http.NewRequestWithContext(ctx, method, s.URL.JoinPath(path).String(), reader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := s.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	b, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode != 200 {
		return nil, &ResponseError{response.StatusCode, b}
	}
	return b, nil
}

func (c *Client) Fleet(ctx context.Context) (json.RawMessage, error) {
	return c.call(ctx, "fleet", nil)
}

func (c *Client) Run(ctx context.Context, t mechanic.Turn, actor string, prev *mechanic.Conversation) (json.RawMessage, error) {
	request := struct {
		mechanic.Turn
		Actor string `json:"actor_id"`
	}{t, actor}
	return c.call(ctx, "run", map[string]any{"turn": request, "context": prev})
}

func (c *Client) Confirm(ctx context.Context, decision mechanic.Decision, actor string) (json.RawMessage, error) {
	return c.call(ctx, "confirm", map[string]any{"decision": decision, "actor_id": actor})
}
