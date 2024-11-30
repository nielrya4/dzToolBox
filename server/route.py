from server import cleanup
from apps import project_browser, init, new_editor, errors, docs, updates


def register_routes(app):
    project_browser.register(app)
    init.register(app)
    new_editor.register(app)
    errors.register(app)
    docs.register(app)
    updates.register(app)
