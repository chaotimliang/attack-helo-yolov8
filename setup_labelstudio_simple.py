#!/usr/bin/env python3
"""
Simple Label Studio Setup Script
This script provides a minimal setup for Label Studio to work with YOLO v8 training.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Setup logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def install_minimal_dependencies():
    """Install minimal dependencies for Label Studio."""
    logger = logging.getLogger(__name__)
    
    # Essential dependencies for Label Studio to run
    essential_deps = [
        'django-environ',
        'django-cors-headers', 
        'django-filter',
        'django-model-utils',
        'django-storages',
        'django-user-agents',
        'django-annoying',
        'django-csp',
        'django-debug-toolbar',
        'django-extensions',
        'django-migration-linter',
        'django-ranged-fileresponse',
        'django-rq',
        'djangorestframework-simplejwt',
        'drf-dynamic-fields',
        'drf-flex-fields',
        'drf-generators',
        'drf-spectacular',
        'label-studio-sdk',
        'lockfile',
        'ordered-set',
        'python-json-logger',
        'rules',
        'tldextract',
        'xmljson',
        'pydantic',
        'pyboxen',
        'sentry-sdk'
    ]
    
    logger.info("Installing essential dependencies...")
    
    for dep in essential_deps:
        try:
            logger.info(f"Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                          check=True, capture_output=True)
            logger.info(f"✓ {dep} installed")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠ Failed to install {dep}: {e}")
            continue
    
    logger.info("Essential dependencies installation completed!")

def create_labelstudio_config():
    """Create Label Studio configuration."""
    logger = logging.getLogger(__name__)
    
    # Create .env file for Label Studio
    env_content = """# Label Studio Configuration
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=./data
LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
LABEL_STUDIO_USERNAME=admin
LABEL_STUDIO_PASSWORD=admin
"""
    
    env_path = Path('.env')
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    logger.info(f"Created Label Studio configuration: {env_path}")

def test_labelstudio():
    """Test if Label Studio can run."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test import
        import label_studio
        logger.info("✓ Label Studio imported successfully")
        
        # Test version
        logger.info(f"✓ Label Studio version: {label_studio.__version__}")
        
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import Label Studio: {e}")
        return False

def create_startup_script():
    """Create a startup script for Label Studio."""
    logger = logging.getLogger(__name__)
    
    # Create startup script
    startup_script = """@echo off
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
"""
    
    startup_path = Path('start_labelstudio.bat')
    with open(startup_path, 'w') as f:
        f.write(startup_script)
    
    logger.info(f"Created startup script: {startup_path}")

def main():
    """Main setup function."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("LABEL STUDIO SIMPLE SETUP")
    logger.info("=" * 60)
    
    # Test current installation
    if test_labelstudio():
        logger.info("Label Studio is already working!")
    else:
        logger.info("Installing minimal dependencies...")
        install_minimal_dependencies()
    
    # Create configuration
    create_labelstudio_config()
    
    # Create startup script
    create_startup_script()
    
    logger.info("=" * 60)
    logger.info("SETUP COMPLETED!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Run: start_labelstudio.bat")
    logger.info("2. Open browser to: http://localhost:8080")
    logger.info("3. Login with: admin / admin")
    logger.info("4. Create a new project")
    logger.info("5. Import the configuration from configs/labelstudio_config.xml")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
