package mechanic

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"time"
)

const TurnLease = 30 * time.Minute
const SessionLifetime = 7 * 24 * time.Hour

var ErrBusy = errors.New("conversation busy")
var ErrMismatch = errors.New("conversation mismatch")

func RandomID(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b)
}

type Conversation struct {
	Actor      string          `json:"actor_id"`
	ID         string          `json:"conversation_id"`
	Vehicle    string          `json:"vehicle_id"`
	Session    string          `json:"session_id"`
	Engine     string          `json:"engine_model"`
	Diagnostic json.RawMessage `json:"diagnostic"`
	Messages   []string        `json:"recent_messages"`
}

type Lease struct{ Actor, ID, Token string }

type Turn struct {
	Language string  `json:"language,omitempty"`
	Vehicle  string  `json:"vehicle_id"`
	Session  string  `json:"session_id"`
	ID       string  `json:"conversation_id"`
	Message  *string `json:"message,omitempty"`
	Command  string  `json:"command,omitempty"`
	// The mechanic confirmed a noisy recording; the worker still refuses bad or failed ones.
	AcceptWarning bool `json:"accept_signal_warning,omitempty"`
	// Inspection result for command record_inspection; the worker validates its fields.
	Inspection json.RawMessage `json:"inspection,omitempty"`
}

var ErrInput = errors.New("invalid agent turn")
var ErrMeasurement = errors.New("unknown measurement")

type Repository interface {
	Actor(context.Context, string) (string, string, error)
	Begin(context.Context, Lease, string, string) (*Conversation, error)
	Finish(Lease, *Conversation) error
}

var ErrConfirmation = errors.New("confirmation required")

type Decision struct {
	Vehicle   string  `json:"vehicle_id"`
	Session   string  `json:"session_id"`
	Draft     string  `json:"draft_id"`
	Version   int     `json:"draft_version"`
	Confirmed *bool   `json:"confirmed"`
	Urgency   *string `json:"urgency,omitempty"`
}

type Agent interface {
	Run(context.Context, Turn, string, *Conversation) (json.RawMessage, error)
	Confirm(context.Context, Decision, string) (json.RawMessage, error)
}

type Service struct {
	Repository   Repository
	Agent        Agent
	Measurements map[[2]string]bool
}

func Measurements(data json.RawMessage) (map[[2]string]bool, error) {
	var fleet struct {
		Vehicles []struct {
			Vehicle  string `json:"vehicle_id"`
			Sessions []struct {
				ID string `json:"session_id"`
			} `json:"sessions"`
		} `json:"vehicles"`
	}
	if err := json.Unmarshal(data, &fleet); err != nil {
		return nil, err
	}
	pairs := make(map[[2]string]bool)
	for _, v := range fleet.Vehicles {
		for _, m := range v.Sessions {
			pairs[[2]string{v.Vehicle, m.ID}] = true
		}
	}
	return pairs, nil
}
