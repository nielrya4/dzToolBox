import os
import app
from datetime import datetime, timedelta
import schedule
import threading
import time


def start_cleanup():
    schedule.every().hour.do(cleanup_job)
    schedule_thread = threading.Thread(target=run_scheduler)
    schedule_thread.start()


def cleanup_job():
    print("Cleaning up files in data folder...")
    # cleanup_folder(app.DATA_FOLDER)
    print("Cleaning up files in upload folder...")
    # cleanup_folder(app.UPLOAD_FOLDER)


def cleanup_folder(folder_path):
    current_time = datetime.now()
    twenty_four_hours_ago = current_time - timedelta(hours=24)

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        file_creation_time = datetime.fromtimestamp(os.path.getctime(filepath))

        if file_creation_time < twenty_four_hours_ago:
            os.remove(filepath)
            print(f"Deleted: {filename}")


def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)
