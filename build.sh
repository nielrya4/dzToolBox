#!/bin/bash
source .env.dztoolbox
# Read password from stdin or use environment variable
if [ -t 0 ]; then
    # Running interactively, use environment variable
    if [ -z "$DZTOOLBOX_PASSWORD" ]; then
        echo "ERROR: DZTOOLBOX_PASSWORD environment variable not set"
        exit 1
    fi
    PASSWORD="$DZTOOLBOX_PASSWORD"
else
    # Password provided via stdin
    read -r PASSWORD
fi

echo "Updating dztoolbox code..."
git pull

echo "Installing Python dependencies..."
# Activate virtual environment and install dependencies
source venv/bin/activate
pip install -r requirements.txt

echo "Checking for dz_lib updates..."
pip install --upgrade dz_lib
deactivate

echo "stopping dztoolbox..."
echo "$PASSWORD" | sudo systemctl stop uwsgi-dztoolbox

echo "updating system files..."
echo "$PASSWORD" | sudo cp ./setup/etc_systemd_system/* /etc/systemd/system/

echo "reloading daemons..."
echo "$PASSWORD" | sudo systemctl daemon-reload

echo "starting up dztoolbox..."
echo "$PASSWORD" | sudo systemctl start uwsgi-dztoolbox

echo "dztoolbox is now up and running"
