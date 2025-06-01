import multiprocessing

# Server socket settings
bind = "0.0.0.0:10000"
backlog = 2048

# Worker processes
workers = 1  # Single worker for memory-intensive tasks
worker_class = 'sync'
worker_connections = 1000
timeout = 600  # Increased to 10 minutes
keepalive = 2

# Process naming
proc_name = 'rite_analysis'

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# SSL
keyfile = None
certfile = None

# Process management
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Memory management
max_requests = 1
max_requests_jitter = 0

# Worker configuration
worker_tmp_dir = '/dev/shm'  # Use memory for temporary files
worker_max_requests = 1  # Restart workers after each request to clear memory
worker_max_requests_jitter = 0

# Misc
reload = False
spew = False
check_config = False 