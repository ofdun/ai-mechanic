package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/ofdun/ai-mechanic/backend/internal/app"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := app.Run(ctx, os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}
