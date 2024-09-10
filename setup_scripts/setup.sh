#!/usr/bin/env bash
curl ifconfig.me

cd ~/
sudo apt update
sudo apt install python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools nginx
sudo apt install python3-venv
git clone https://www.github.com/nielrya4/dztoolbox.git
cd dztoolbox
python3.10 -m venv venv
source venv/bin/activate
pip install wheel
pip install -r requirements.txt
sudo mv setup_scripts/dztoolbox.service /etc/systemd/system/
sudo chgrp www-data ~/
sudo systemctl start dztoolbox
sudo systemctl enable dztoolbox
sudo systemctl stop dztoolbox
sudo mv setup_scripts/dztoolbox /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/dztoolbox /etc/nginx/sites-enabled
sudo unlink /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
sudo ufw allow 'Nginx Full'
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d dztoolbox.com -d www.dztoolbox.com
sudo ufw delete allow 'Nginx HTTP'

