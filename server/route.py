from server import cleanup
from apps import project_browser, init, editor, errors, docs


def register_routes(app):
    project_browser.register(app)
    init.register(app)
    editor.register(app)
    errors.register(app)
    docs.register(app)
