#!/bin/bash

# Start Celery worker for dzToolBox background tasks

echo "Starting Celery worker..."
echo "Make sure Redis is running: redis-cli ping"
echo ""

cd "$(dirname "$0")"

celery -A celery_tasks worker --loglevel=info --pool=solo

# Use --pool=solo to avoid forking (Julia doesn't handle fork() well)
# This runs tasks sequentially in the main process, avoiding segfaults
