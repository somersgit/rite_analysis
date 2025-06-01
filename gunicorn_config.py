import multiprocessing

# Server socket settings
bind = "0.0.0.0:10000"
backlog = 2048

# Worker processes
workers = 1  # Single worker for memory-intensive tasks
worker_class = 'sync'
worker_connections = 1000
timeout = 300  # 5 minutes timeout
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
max_requests = 1000
max_requests_jitter = 50

# Misc
reload = False
spew = False
check_config = False 