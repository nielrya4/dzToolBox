def check(file_format: str=None, formats: [str]=None, accepted_formats: [str]=None):
    if file_format is None:
        if formats is None or len(formats) == 0:
            return False
    else:
        formats = [file_format]
    for file_format in formats:
        if file_format not in accepted_formats:
            return False
    return True