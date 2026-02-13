"""
Background job management for long-running tasks
"""

import threading
import time
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
import traceback


@dataclass
class BackgroundJob:
    """Represents a background job"""
    job_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    result: Any = None
    error: Optional[str] = None
    progress: int = 0  # 0-100
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BackgroundJobManager:
    """Manages background jobs"""

    def __init__(self):
        self.jobs: Dict[str, BackgroundJob] = {}
        self.lock = threading.Lock()

    def create_job(self, job_id: str) -> BackgroundJob:
        """Create a new job"""
        job = BackgroundJob(job_id=job_id, status='pending')
        with self.lock:
            self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[BackgroundJob]:
        """Get job by ID"""
        with self.lock:
            return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        """Update job fields"""
        with self.lock:
            if job_id in self.jobs:
                for key, value in kwargs.items():
                    setattr(self.jobs[job_id], key, value)

    def run_job(self, job_id: str, func: Callable, *args, **kwargs):
        """Run a function as a background job"""
        def _run():
            job = self.get_job(job_id)
            if not job:
                return

            try:
                self.update_job(job_id, status='running', started_at=datetime.now())
                result = func(*args, **kwargs)
                self.update_job(
                    job_id,
                    status='completed',
                    result=result,
                    completed_at=datetime.now(),
                    progress=100
                )
            except Exception as e:
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                self.update_job(
                    job_id,
                    status='failed',
                    error=error_msg,
                    completed_at=datetime.now()
                )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def cleanup_old_jobs(self, max_age_seconds: int = 3600):
        """Remove jobs older than max_age_seconds"""
        now = datetime.now()
        with self.lock:
            expired = [
                job_id for job_id, job in self.jobs.items()
                if (now - job.created_at).total_seconds() > max_age_seconds
            ]
            for job_id in expired:
                del self.jobs[job_id]


# Global job manager instance
job_manager = BackgroundJobManager()
