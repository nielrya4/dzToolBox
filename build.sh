echo "Updating dztoolbox code..."
git pull
echo "stopping dztoolbox..."
sudo systemctl stop dztoolbox
echo "reloading daemons..."
sudo systemctl daemon-reload
echo "starting up dztoolbox..."
sudo systemctl start dztoolbox
echo "restarting nginx..."
sudo systemctl restart nginx
echo "dztoolbox is now up and running"
