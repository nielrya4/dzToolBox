from apps import project_browser, init, new_editor, errors, docs, updates, databases
from api import account, outputs, data

def register_routes(app):
    project_browser.register(app)
    init.register(app)
    new_editor.register(app)
    errors.register(app)
    docs.register(app)
    updates.register(app)
    databases.register(app)
    account.register(app)
    outputs.register(app)
    data.register(app)