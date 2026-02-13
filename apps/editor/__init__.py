"""
Editor module - contains all editor-related apps
"""

def register(app):
    """Register all editor sub-apps"""
    from . import dz_grainalyzer
    from . import spreadsheet
    from . import outputs
    from . import maps

    # Register each sub-app
    dz_grainalyzer.register(app)
    spreadsheet.register(app)
    outputs.register(app)
    maps.register(app)
