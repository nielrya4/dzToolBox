from flask_login import current_user
import dzToolBox as APP
from sqlalchemy.orm import selectinload


def get_file(file_id):
    file = APP.CodeFile.query.filter_by(user_id=current_user.id, id=file_id).first_or_404()
    return file


def get_all_files(limit=None, offset=None):
    query = APP.CodeFile.query.filter_by(user_id=current_user.id).order_by(APP.CodeFile.id.desc())
    if limit:
        query = query.limit(limit)
    if offset:
        query = query.offset(offset)
    return query.all()


def get_files_count():
    return APP.CodeFile.query.filter_by(user_id=current_user.id).count()


def new_file(title, content):
    file = APP.CodeFile(title=title, content=content, author=current_user)
    APP.db.session.add(file)
    APP.db.session.commit()
    return file


def delete_file(file_id):
    try:
        result = APP.CodeFile.query.filter_by(user_id=current_user.id, id=file_id).delete()
        if result > 0:
            APP.db.session.commit()
            return 0
        else:
            APP.db.session.rollback()
            return 1
    except Exception as e:
        APP.db.session.rollback()
        print(f"Error deleting file: {e}")
        return 1


def delete_multiple_files(file_ids):
    try:
        result = APP.CodeFile.query.filter(
            APP.CodeFile.user_id == current_user.id,
            APP.CodeFile.id.in_(file_ids)
        ).delete(synchronize_session=False)
        APP.db.session.commit()
        return result
    except Exception as e:
        APP.db.session.rollback()
        print(f"Error deleting multiple files: {e}")
        return 0


def write_file(file_id, new_content):
    try:
        result = APP.CodeFile.query.filter_by(
            user_id=current_user.id, 
            id=file_id
        ).update({'content': new_content})
        if result > 0:
            APP.db.session.commit()
            return 0
        else:
            APP.db.session.rollback()
            return 1
    except Exception as e:
        APP.db.session.rollback()
        print(f"Error modifying file: {e}")
        return 1


def rename_file(file_id, new_title):
    try:
        result = APP.CodeFile.query.filter_by(
            user_id=current_user.id, 
            id=file_id
        ).update({'title': new_title})
        if result > 0:
            APP.db.session.commit()
            return 0
        else:
            APP.db.session.rollback()
            return 1
    except Exception as e:
        APP.db.session.rollback()
        print(f"Error modifying file: {e}")
        return 1
