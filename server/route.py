from server import account, cleanup
from apps import project_browser, init


def register_routes(app):
    account.register(app)
    project_browser.register(app)
    init.register(app)
