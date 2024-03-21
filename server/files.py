from werkzeug.utils import secure_filename
import os
import app as APP


def upload_file(file):
    if file:
        filename = f"{secure_filename(file.filename)}"
        file_path = os.path.join("temp", filename)
        file.save(file_path)
        return file_path
    return None
