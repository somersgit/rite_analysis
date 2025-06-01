import multiprocessing

# Server socket settings
bind = "0.0.0.0:10000"
backlog = 1024  # Reduced from 2048

# Worker processes
workers = 1
worker_class = 'sync'
worker_connections = 100  # Reduced from 1000
timeout = 900  # Increased to 15 minutes
keepalive = 2

# Process naming
proc_name = 'rite_analysis'

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'debug'  # Changed to debug for more information

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
worker_tmp_dir = '/dev/shm'
worker_max_requests = 1
worker_max_requests_jitter = 0

# Memory limits (in bytes)
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Misc
reload = False
spew = False
check_config = False 