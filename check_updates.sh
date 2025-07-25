#!/bin/bash

REPO_DIR="/home/dztoolbox/dztoolbox"
REPO_URL="https://github.com/nielrya4/dztoolbox.git"

cd "$REPO_DIR" || exit

git fetch

LOCAL_HASH=$(git rev-parse HEAD)
REMOTE_HASH=$(git rev-parse origin/main)

if [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
    echo "New updates found, running build.sh"
    /home/dztoolbox/dztoolbox/build.sh
else
    echo "No updates found."
fi