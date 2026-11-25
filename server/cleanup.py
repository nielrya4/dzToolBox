import os
import dzToolBox as APP
from datetime import datetime, timezone, timedelta
from sqlalchemy import or_
import schedule
import threading
import time


def start_cleanup():
    mst_offset = -7
    mst_time = (datetime.now(timezone.utc) + timedelta(hours=mst_offset)).replace(hour=23, minute=59, second=59, microsecond=0).time()
    schedule.every().day.at(mst_time.strftime('%H:%M:%S')).do(cleanup_job)
    schedule_thread = threading.Thread(target=run_scheduler)
    schedule_thread.start()


def cleanup_job():
    with APP.app.app_context():
        try:
            # Only guests older than 12 hours (or of unknown age), so nobody loses a session in progress.
            cutoff = datetime.now(timezone.utc) - timedelta(hours=12)
            guest_users = APP.User.query.filter(
                APP.User.is_guest.is_(True),
                or_(APP.User.created_at.is_(None), APP.User.created_at < cutoff)
            ).all()
            
            if not guest_users:
                print("No guest accounts to clean up")
                return
                
            guest_user_ids = [user.id for user in guest_users]
            print(f"Found {len(guest_users)} guest accounts to clean up")
            
            deleted_files = APP.CodeFile.query.filter(
                APP.CodeFile.user_id.in_(guest_user_ids)
            ).delete(synchronize_session=False)
            print(f"Deleted {deleted_files} files from guest accounts")
            
            deleted_users = APP.User.query.filter(
                APP.User.id.in_(guest_user_ids)
            ).delete(synchronize_session=False)
            print(f"Deleted {deleted_users} guest accounts")
            
            APP.db.session.commit()
            print("Cleaned Up Guest Accounts Successfully")
            
        except Exception as e:
            APP.db.session.rollback()
            print(f"Error during cleanup: {e}")
            raise


def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)
