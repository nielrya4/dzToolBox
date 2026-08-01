import os
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties


# Function to return a list of system fonts
def get_sys_fonts():
    """
    Returns a list of available system fonts.
    """
    return fm.get_fontconfig_fonts()


# Function to load a font from a file path
def get_font(font_path: str) -> FontProperties:
    """
    Loads a font from the specified font file path.

    :param font_path: Path to the font file (e.g., .ttf or .otf).
    :return: FontProperties object for the font.
    :raises ValueError: If the file path is invalid or the file doesn't exist.
    :raises TypeError: If font_path is not a valid string or path-like object.
    """
    if not isinstance(font_path, (str, bytes, os.PathLike)):
        raise TypeError(f"font_path should be a valid string or path-like object, got {type(font_path).__name__}")

    if not os.path.isfile(font_path):
        raise ValueError(f"Font file {font_path} does not exist.")

    return FontProperties(fname=font_path)


# Function to return the default system font
def get_default_font() -> FontProperties:
    """
    Returns the default system font.
    This is useful if no custom font is provided.

    :return: FontProperties object for the default font.
    """
    default_font = FontProperties()  # Gets the default font properties from matplotlib
    return default_font


# Optional function to list installed fonts (useful for debugging or selection)
def list_available_fonts() -> list:
    """
    Lists available fonts on the system.

    :return: List of font names (strings).
    """
    font_files = get_sys_fonts()
    available_fonts = [fm.FontProperties(fname=font).get_name() for font in font_files]
    return sorted(available_fonts)


# Debug function to display the default font
def print_default_font_info():
    """
    Prints the name and properties of the default font for debugging purposes.
    """
    default_font = get_default_font()
    print(f"Default font: {default_font.get_name()}")
    print(f"Default font size: {default_font.get_size()}")
