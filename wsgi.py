from dzToolBox import app, db
from flask_cloudflared import run_with_cloudflared

run_with_cloudflared(app)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        app.run()
