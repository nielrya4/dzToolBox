cd ../..
IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
open http://$IP:5000
uwsgi --socket 0.0.0.0:5000 --protocol=http -w wsgi:app