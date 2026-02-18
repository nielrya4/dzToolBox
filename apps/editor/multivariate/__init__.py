"""
Multivariate editor sub-apps
"""


def register(app):
    from . import spreadsheet, outputs
    spreadsheet.register(app)
    outputs.register(app)
