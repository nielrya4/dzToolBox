#!/usr/bin/env python3
"""
Management script for dzToolBox
Handles Celery worker lifecycle and other management tasks
"""

import os
import sys
import signal
import subprocess
import time
import psutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
PID_FILE = PROJECT_ROOT / "celery_worker.pid"
LOG_FILE = PROJECT_ROOT / "celery_worker.log"


def get_worker_process():
    """Find the running Celery worker process"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'celery' in ' '.join(cmdline) and 'celery_tasks' in ' '.join(cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def is_worker_running():
    """Check if Celery worker is running"""
    # Check PID file first
    if PID_FILE.exists():
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                cmdline = ' '.join(proc.cmdline())
                if 'celery' in cmdline and 'celery_tasks' in cmdline:
                    return True
        except (ValueError, psutil.NoSuchProcess):
            # PID file exists but process doesn't, clean it up
            PID_FILE.unlink()

    # Fall back to searching for process
    return get_worker_process() is not None


def check_redis():
    """Check if Redis is running"""
    try:
        result = subprocess.run(
            ['redis-cli', 'ping'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0 and 'PONG' in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_worker():
    """Start the Celery worker"""
    if is_worker_running():
        print("❌ Celery worker is already running")
        return False

    print("🔍 Checking Redis connection...")
    if not check_redis():
        print("❌ Redis is not running. Please start Redis first:")
        print("   sudo systemctl start redis")
        print("   OR")
        print("   redis-server")
        return False
    print("✓ Redis is running")

    print("🚀 Starting Celery worker...")

    # Start the worker as a background process
    with open(LOG_FILE, 'w') as log:
        proc = subprocess.Popen(
            [
                'celery', '-A', 'celery_tasks',
                'worker',
                '--loglevel=info',
                '--pool=solo',  # Julia doesn't handle fork() well
                '--logfile', str(LOG_FILE)
            ],
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )

    # Wait a bit and check if it's still running
    time.sleep(2)
    if proc.poll() is not None:
        print("❌ Failed to start Celery worker. Check logs at:", LOG_FILE)
        return False

    # Save PID
    with open(PID_FILE, 'w') as f:
        f.write(str(proc.pid))

    print(f"✓ Celery worker started (PID: {proc.pid})")
    print(f"📝 Logs: {LOG_FILE}")
    return True


def stop_worker():
    """Stop the Celery worker"""
    if not is_worker_running():
        print("❌ Celery worker is not running")
        return False

    print("🛑 Stopping Celery worker...")

    # Try to get process from PID file first
    proc = None
    if PID_FILE.exists():
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            proc = psutil.Process(pid)
        except (ValueError, psutil.NoSuchProcess):
            pass

    # Fall back to searching for process
    if proc is None:
        proc = get_worker_process()

    if proc is None:
        print("❌ Could not find worker process")
        # Clean up PID file if it exists
        if PID_FILE.exists():
            PID_FILE.unlink()
        return False

    try:
        # Send SIGTERM for graceful shutdown
        proc.terminate()

        # Wait up to 10 seconds for graceful shutdown
        try:
            proc.wait(timeout=10)
            print("✓ Celery worker stopped gracefully")
        except psutil.TimeoutExpired:
            # Force kill if graceful shutdown failed
            print("⚠ Graceful shutdown timed out, forcing kill...")
            proc.kill()
            proc.wait(timeout=5)
            print("✓ Celery worker killed")

        # Clean up PID file
        if PID_FILE.exists():
            PID_FILE.unlink()

        return True
    except psutil.NoSuchProcess:
        print("✓ Worker process already stopped")
        if PID_FILE.exists():
            PID_FILE.unlink()
        return True


def restart_worker():
    """Restart the Celery worker"""
    print("🔄 Restarting Celery worker...")
    stop_worker()
    time.sleep(1)
    return start_worker()


def status_worker():
    """Show Celery worker status"""
    print("📊 Celery Worker Status")
    print("=" * 50)

    # Check Redis
    redis_status = "✓ Running" if check_redis() else "❌ Not running"
    print(f"Redis:  {redis_status}")

    # Check Worker
    if is_worker_running():
        proc = get_worker_process()
        if proc:
            cpu = proc.cpu_percent(interval=0.1)
            mem = proc.memory_info().rss / 1024 / 1024  # MB
            uptime = time.time() - proc.create_time()
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)

            print(f"Worker: ✓ Running (PID: {proc.pid})")
            print(f"  CPU:    {cpu:.1f}%")
            print(f"  Memory: {mem:.1f} MB")
            print(f"  Uptime: {hours}h {minutes}m")
        else:
            print("Worker: ✓ Running")
    else:
        print("Worker: ❌ Not running")

    print(f"\nLog file: {LOG_FILE}")
    print(f"PID file: {PID_FILE}")


def show_logs(lines=50, follow=False):
    """Show Celery worker logs"""
    if not LOG_FILE.exists():
        print(f"❌ Log file not found: {LOG_FILE}")
        return

    if follow:
        print(f"📝 Following logs from {LOG_FILE} (Ctrl+C to stop)...")
        try:
            subprocess.run(['tail', '-f', str(LOG_FILE)])
        except KeyboardInterrupt:
            print("\n✓ Stopped following logs")
    else:
        print(f"📝 Last {lines} lines from {LOG_FILE}:")
        print("=" * 50)
        subprocess.run(['tail', f'-n{lines}', str(LOG_FILE)])


def print_help():
    """Print help message"""
    print("""
dzToolBox Management Script

Usage:
    python manage.py <command> [options]

Commands:
    start       Start the Celery worker
    stop        Stop the Celery worker
    restart     Restart the Celery worker
    status      Show worker status
    logs        Show recent logs (default: 50 lines)
    logs -f     Follow logs in real-time
    help        Show this help message

Examples:
    python manage.py start
    python manage.py restart
    python manage.py logs -f
    """)


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'start':
        success = start_worker()
        sys.exit(0 if success else 1)

    elif command == 'stop':
        success = stop_worker()
        sys.exit(0 if success else 1)

    elif command == 'restart':
        success = restart_worker()
        sys.exit(0 if success else 1)

    elif command == 'status':
        status_worker()
        sys.exit(0)

    elif command == 'logs':
        follow = '-f' in sys.argv or '--follow' in sys.argv
        show_logs(follow=follow)
        sys.exit(0)

    elif command == 'help' or command == '--help' or command == '-h':
        print_help()
        sys.exit(0)

    else:
        print(f"❌ Unknown command: {command}")
        print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
