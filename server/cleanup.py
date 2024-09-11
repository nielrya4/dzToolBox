import os
import app as APP
from datetime import datetime, timezone, timedelta
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
        for user in APP.User.query.all():
            print(user.username)
            if user.username.endswith("_guest"):
                APP.CodeFile.query.filter_by(user_id=user.id).delete()
                APP.db.session.delete(user)
                APP.db.session.commit()
                print(f"Deleted Account: {user.username}")
    print("Cleaned Up Guest Accounts")


def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)
