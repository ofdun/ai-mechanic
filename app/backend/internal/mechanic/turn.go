package mechanic

import (
	"strings"
	"unicode/utf8"
)

const maxMessageRunes = 8000

func (t *Turn) normalize() error {
	t.Vehicle = strings.TrimSpace(t.Vehicle)
	t.Session = strings.TrimSpace(t.Session)
	t.ID = strings.TrimSpace(t.ID)
	if t.Message != nil {
		clean := strings.TrimSpace(*t.Message)
		if clean == "" || utf8.RuneCountInString(clean) > maxMessageRunes {
			return ErrInput
		}
		t.Message = &clean
	}
	switch t.Command {
	case "", "auto":
		if t.Message == nil {
			return ErrInput
		}
	case "diagnose", "draft_ticket", "get_diagnostic_history":
	case "record_inspection":
		if len(t.Inspection) == 0 {
			return ErrInput
		}
	default:
		return ErrInput
	}
	if t.Vehicle == "" || t.Session == "" {
		return ErrInput
	}
	return nil
}
