package mechanic

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestInspectionTurnNeedsItsPayload(t *testing.T) {
	cases := []struct {
		turn Turn
		err  error
	}{
		{Turn{Vehicle: "V", Session: "S", Command: "record_inspection"}, ErrInput},
		{Turn{Vehicle: "V", Session: "S", Command: "record_inspection", Inspection: json.RawMessage(`{"outcome":"confirmed"}`)}, nil},
		{Turn{Vehicle: "V", Session: "S", Command: "get_diagnostic_history"}, nil},
		{Turn{Vehicle: "V", Session: "S", Command: "delete_everything"}, ErrInput},
	}
	for _, c := range cases {
		if err := c.turn.normalize(); !errors.Is(err, c.err) {
			t.Errorf("%s: got %v, want %v", c.turn.Command, err, c.err)
		}
	}
}
