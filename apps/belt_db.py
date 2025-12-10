from flask import render_template

def register(app):
    @app.route('/belt-db', methods=['GET'])
    def belt_db():
        return render_template("belt_db/belt_db.html")
