#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
REPO_URL="https://github.com/nielrya4/dztoolbox.git"
LOG_FILE="${REPO_DIR}/update.log"

# Function to log with timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_message "Starting update check..."

cd "$REPO_DIR" || {
    log_message "ERROR: Failed to change to repository directory: $REPO_DIR"
    exit 1
}

# Fetch latest changes
log_message "Fetching latest changes from remote..."
if ! git fetch; then
    log_message "ERROR: Failed to fetch from remote repository"
    exit 1
fi

LOCAL_HASH=$(git rev-parse HEAD)
REMOTE_HASH=$(git rev-parse origin/master)

log_message "Local hash: $LOCAL_HASH"
log_message "Remote hash: $REMOTE_HASH"

if [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
    log_message "New updates found, running build.sh"
    if echo "d33zt00ls" | "${REPO_DIR}/build.sh"; then
        log_message "Update completed successfully"
    else
        log_message "ERROR: Update failed"
        exit 1
    fi
else
    log_message "No updates found."
fi