from server import account, cleanup, database
from apps import editor, init


def register_routes(app):
    account.register(app)
    editor.register(app)
    init.register(app)
