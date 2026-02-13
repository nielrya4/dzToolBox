# Celery + Redis Setup for Background Tasks

This guide will help you set up Celery with Redis for handling long-running tensor factorization tasks.

## Prerequisites

1. **Redis** - In-memory data store used as message broker
2. **Celery** - Distributed task queue

## Installation

### 1. Install Redis

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**macOS (with Homebrew):**
```bash
brew install redis
brew services start redis
```

**Windows:**
Download from https://github.com/microsoftarchive/redis/releases or use WSL

### 2. Install Python Dependencies

```bash
pip install celery redis
```

Or add to your requirements.txt:
```
celery==5.3.4
redis==5.0.1
```

### 3. Verify Redis is Running

```bash
redis-cli ping
```

Should return: `PONG`

## Running the Application

You need **TWO** separate processes running (Redis + Flask), plus the Celery worker managed via the management script.

### Quick Start (Recommended)

**Terminal 1: Start Celery Worker**
```bash
cd /home/ryan/PycharmProjects/dzToolBox
python manage.py start
```

**Terminal 2: Flask Web Server**
```bash
cd /home/ryan/PycharmProjects/dzToolBox
python dzToolBox.py
```

### Management Commands

The `manage.py` script provides easy worker management:

```bash
# Start the worker
python manage.py start

# Stop the worker
python manage.py stop

# Restart the worker
python manage.py restart

# Check worker status
python manage.py status

# View logs (last 50 lines)
python manage.py logs

# Follow logs in real-time
python manage.py logs -f

# Show help
python manage.py help
```

### Manual Method (Alternative)

If you prefer to run the worker manually:

**Terminal 1: Flask Web Server**
```bash
cd /home/ryan/PycharmProjects/dzToolBox
python dzToolBox.py
```

**Terminal 2: Celery Worker**
```bash
cd /home/ryan/PycharmProjects/dzToolBox
celery -A celery_tasks worker --loglevel=info --pool=solo
```

**Terminal 3: Redis Server (if not running as service)**
```bash
redis-server
```

## Configuration

By default, the application uses:
- Redis URL: `redis://localhost:6379/0`

To customize, set environment variables:
```bash
export CELERY_BROKER_URL='redis://localhost:6379/0'
export CELERY_RESULT_BACKEND='redis://localhost:6379/0'
```

## Testing the Setup

1. Start all three processes (Flask, Celery worker, Redis)
2. Open dzToolBox in browser
3. Go to Outputs tab
4. Click "Tensor Factorization"
5. Select samples and click "Factorize"
6. Check Celery worker terminal - you should see task starting
7. Wait for completion - outputs will auto-save

## Monitoring Tasks

### Check Celery worker status:
```bash
celery -A celery_tasks inspect active
```

### Check task result:
```bash
celery -A celery_tasks result <task_id>
```

### Purge all tasks:
```bash
celery -A celery_tasks purge
```

## Production Considerations

For production deployment:

1. **Use a process manager** (systemd, supervisor) to keep Celery running
2. **Scale workers**: Run multiple workers for parallel processing
   ```bash
   celery -A celery_tasks worker --concurrency=4
   ```
3. **Monitor with Flower** (Celery monitoring tool):
   ```bash
   pip install flower
   celery -A celery_tasks flower
   ```
   Then visit http://localhost:5555

4. **Configure Redis persistence** if you need task results to survive restarts

5. **Set resource limits** in celery_app.py (already configured):
   - task_time_limit: 1 hour max
   - result_expires: 1 hour

## Troubleshooting

**"Celery is not configured" error:**
- Make sure Celery worker is running
- Check that celery and redis packages are installed

**Tasks stuck in PENDING:**
- Check Redis is running: `redis-cli ping`
- Verify Celery worker is running and connected
- Check worker logs for errors

**Out of memory:**
- Reduce number of concurrent workers
- Increase system swap space
- Use a machine with more RAM for large tensor factorizations

**Julia package installation hangs:**
- First run may take 5-10 minutes to install Julia packages
- This only happens once per environment
- Subsequent runs will be faster

## Deployment and Updates

### Production Deployment

The production environment runs both the Flask app (via uWSGI) and the Celery worker as systemd services. The deployment is fully automated via `build.sh`.

**Systemd Services:**
- `uwsgi-dztoolbox.service` - Flask web application
- `celery-dztoolbox.service` - Celery background worker

**Deployment Process:**

```bash
# From your local machine or CI/CD
cd /home/ryan/PycharmProjects/dzToolBox
./build.sh
```

The `build.sh` script automatically:
1. Pulls latest code from git
2. Updates Python dependencies
3. Upgrades dz_lib
4. Stops both services (uWSGI and Celery)
5. Updates systemd service files
6. Reloads systemd daemon
7. Starts both services
8. Displays service status

**Manual Service Management:**

```bash
# Restart both services
sudo systemctl restart uwsgi-dztoolbox
sudo systemctl restart celery-dztoolbox

# Check status
sudo systemctl status uwsgi-dztoolbox
sudo systemctl status celery-dztoolbox

# View logs
sudo journalctl -u uwsgi-dztoolbox -f
sudo journalctl -u celery-dztoolbox -f

# Or view log files directly
sudo tail -f /var/log/uwsgi-dztoolbox.log
sudo tail -f /var/log/celery-dztoolbox.log

# Enable services on boot
sudo systemctl enable uwsgi-dztoolbox
sudo systemctl enable celery-dztoolbox
```

**First-Time Setup on Production Server:**

After copying the code to `/home/dztoolbox/dztoolbox`:

```bash
# Copy systemd service files
sudo cp ./setup/etc_systemd_system/*.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable uwsgi-dztoolbox
sudo systemctl enable celery-dztoolbox

# Start services
sudo systemctl start uwsgi-dztoolbox
sudo systemctl start celery-dztoolbox

# Check status
sudo systemctl status uwsgi-dztoolbox
sudo systemctl status celery-dztoolbox
```

### Development Environment

For local development, you can use the `manage.py` script for easier worker management:

```bash
# Start worker
python manage.py start

# Stop worker
python manage.py stop

# Restart worker
python manage.py restart

# Check status
python manage.py status

# View logs
python manage.py logs -f
```

**Note:** The `manage.py` script is for development only. Production uses systemd services managed by `build.sh`.
