"""
Editor module - contains all editor-related apps
"""


def register(app):
    """Register all editor sub-apps"""
    from . import univariate, multivariate, maps

    univariate.register(app)
    multivariate.register(app)
    maps.register(app)
