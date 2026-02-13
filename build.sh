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

echo "Ensuring celery, redis, and juliacall are installed for background tasks..."
pip install celery redis juliacall

echo "Pre-installing Julia packages for tensor factorization..."
python3 -c "
try:
    from utils.tensor_factorization import initialize_julia_packages
    print('Installing MatrixTensorFactor and KernelDensity packages...')
    initialize_julia_packages()
    print('Julia packages installed successfully!')
except Exception as e:
    print(f'Warning: Could not pre-install Julia packages: {e}')
"

echo "Checking for dz_lib updates..."
pip install --upgrade dz_lib
deactivate

echo "stopping dztoolbox services..."
echo "$PASSWORD" | sudo -S systemctl stop uwsgi-dztoolbox
echo "$PASSWORD" | sudo -S systemctl stop celery-dztoolbox

echo "updating system files..."
echo "$PASSWORD" | sudo -S cp ./setup/etc_systemd_system/* /etc/systemd/system/

echo "reloading daemons..."
echo "$PASSWORD" | sudo -S systemctl daemon-reload

echo "starting up dztoolbox services..."
echo "$PASSWORD" | sudo -S systemctl start uwsgi-dztoolbox
echo "$PASSWORD" | sudo -S systemctl start celery-dztoolbox

echo "checking service status..."
echo "$PASSWORD" | sudo -S systemctl status uwsgi-dztoolbox --no-pager
echo "$PASSWORD" | sudo -S systemctl status celery-dztoolbox --no-pager

echo "dztoolbox is now up and running"
