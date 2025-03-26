echo "Updating dztoolbox code..."
git pull
echo "stopping dztoolbox..."
sudo systemctl stop uwsgi-dztoolbox
sudo systemctl stop cloudflared
echo "reloading daemons..."
sudo systemctl daemon-reload
echo "starting up dztoolbox..."
sudo systemctl start uwsgi-dztoolbox
sudo systemctl start cloudflared
echo "dztoolbox is now up and running"
