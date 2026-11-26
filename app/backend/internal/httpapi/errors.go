package httpapi

import (
	"errors"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/ofdun/ai-mechanic/backend/internal/agent"
	"github.com/ofdun/ai-mechanic/backend/internal/mechanic"
)

func fail(c *gin.Context, status int, code, msg string, retryable bool) {
	c.JSON(status, gin.H{"detail": gin.H{"code": code, "message": msg, "retryable": retryable}})
}

func problem(c *gin.Context, err error) {
	var remote *agent.ResponseError
	switch {
	case errors.As(err, &remote):
		c.Data(remote.Status, jsonType, remote.Body)
	case errors.Is(err, mechanic.ErrConfirmation):
		fail(c, http.StatusUnprocessableEntity, "CONFIRMATION_REQUIRED", "Нужно явное подтверждение механика.", false)
	case errors.Is(err, mechanic.ErrInput):
		fail(c, http.StatusUnprocessableEntity, "INVALID_INPUT", "Проверьте параметры запроса.", false)
	case errors.Is(err, mechanic.ErrMeasurement):
		fail(c, http.StatusNotFound, "VEHICLE_OR_SESSION_NOT_FOUND", "Замер не найден для выбранной машины.", false)
	case errors.Is(err, mechanic.ErrBusy):
		fail(c, http.StatusConflict, "CONVERSATION_BUSY", "В этом диалоге ещё готовится ответ.", true)
	case errors.Is(err, mechanic.ErrMismatch):
		fail(c, http.StatusConflict, "CONTEXT_MISMATCH", "Для выбранной машины или проверки нужен новый диалог.", false)
	default:
		log.Printf("request failed: %v", err)
		fail(c, http.StatusServiceUnavailable, "SERVICE_UNAVAILABLE", "Сервис временно недоступен.", true)
	}
}
