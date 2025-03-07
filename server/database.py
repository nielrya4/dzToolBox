from flask_login import current_user
import dzToolBox as APP


def get_file(file_id):
    file = APP.CodeFile.query.get_or_404(file_id)
    return file


def get_all_files():
    return APP.CodeFile.query.filter_by(user_id=current_user.id).all()


def new_file(title, content):
    file = APP.CodeFile(title=title, content=content, author=current_user)
    APP.db.session.add(file)
    APP.db.session.commit()
    return file


def delete_file(file_id):
    try:
        file = APP.CodeFile.query.filter_by(user_id=current_user.id, id=file_id).first()
        if file:
            APP.db.session.delete(file)
            APP.db.session.commit()
            return 0
        else:
            return 1
    except Exception as e:
        print(f"Error deleting file: {e}")
        return 1


def write_file(file_id, new_content):
    try:
        file = APP.CodeFile.query.filter_by(user_id=current_user.id, id=file_id).first()
        if file:
            file.content = new_content
            APP.db.session.commit()
            return 0
        else:
            return 1
    except Exception as e:
        print(f"Error modifying file: {e}")
        return 1


def rename_file(file_id, new_title):
    try:
        file = APP.CodeFile.query.filter_by(user_id=current_user.id, id=file_id).first()
        if file:
            file.title = new_title
            APP.db.session.commit()
            return 0
        else:
            return 1
    except Exception as e:
        print(f"Error modifying file: {e}")
        return 1
