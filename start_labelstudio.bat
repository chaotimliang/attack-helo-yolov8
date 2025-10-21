@echo off
echo Starting Label Studio...
echo.
echo Label Studio will be available at: http://localhost:8080
echo Default credentials: admin / admin
echo.
echo Press Ctrl+C to stop the server
echo.

python -c "
import os
import sys
import subprocess
import signal

def signal_handler(sig, frame):
    print('\\nStopping Label Studio...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Set environment variables
os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED'] = 'true'
os.environ['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT'] = './data'
os.environ['LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK'] = 'true'

# Start Label Studio
try:
    subprocess.run(['label-studio', 'start', '--host', 'localhost', '--port', '8080', '--no-browser'])
except KeyboardInterrupt:
    print('\\nLabel Studio stopped.')
"
