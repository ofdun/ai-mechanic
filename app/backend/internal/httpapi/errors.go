package httpapi

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"

	"github.com/ofdun/ai-mechanic/backend/internal/agent"
	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

func fail(w http.ResponseWriter, status int, code, msg string, retryable bool) {
	b, _ := json.Marshal(map[string]any{"detail": map[string]any{"code": code, "message": msg, "retryable": retryable}})
	reply(w, status, b)
}

func problem(w http.ResponseWriter, err error) {
	var remote *agent.ResponseError
	switch {
	case errors.As(err, &remote):
		reply(w, remote.Status, remote.Body)
	case errors.Is(err, mechanic.ErrConfirmation):
		fail(w, http.StatusUnprocessableEntity, "CONFIRMATION_REQUIRED", "Нужно явное подтверждение механика.", false)
	case errors.Is(err, mechanic.ErrInput):
		fail(w, http.StatusUnprocessableEntity, "INVALID_INPUT", "Проверьте параметры запроса.", false)
	case errors.Is(err, mechanic.ErrMeasurement):
		fail(w, http.StatusNotFound, "VEHICLE_OR_SESSION_NOT_FOUND", "Замер не найден для выбранной машины.", false)
	case errors.Is(err, mechanic.ErrBusy):
		fail(w, http.StatusConflict, "CONVERSATION_BUSY", "В этом диалоге ещё готовится ответ.", true)
	case errors.Is(err, mechanic.ErrMismatch):
		fail(w, http.StatusConflict, "CONTEXT_MISMATCH", "Для выбранной машины или проверки нужен новый диалог.", false)
	default:
		log.Printf("request failed: %v", err)
		fail(w, http.StatusServiceUnavailable, "SERVICE_UNAVAILABLE", "Сервис временно недоступен.", true)
	}
}
