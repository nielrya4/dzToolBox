cd ../..
sudo ufw allow 5000
IP=$(hostname -I | awk '{print $1}')
xdg-open http://$IP:5000
uwsgi --socket 0.0.0.0:5000 --protocol=http -w wsgi:app
