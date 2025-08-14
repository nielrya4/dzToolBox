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
sudo systemctl stop uwsgi-dztoolbox
sudo systemctl stop cloudflared
echo "reloading daemons..."
sudo systemctl daemon-reload
echo "starting up dztoolbox..."
sudo systemctl start uwsgi-dztoolbox
sudo systemctl start cloudflared
echo "dztoolbox is now up and running"
