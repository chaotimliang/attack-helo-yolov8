#!/bin/bash

echo "Installing YOLO v8 Training Project Dependencies..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "Python found. Installing dependencies..."
echo

# Install requirements
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo
    echo "Error: Failed to install dependencies"
    echo "Please check your internet connection and try again"
    exit 1
fi

echo
echo "Installation completed successfully!"
echo
echo "Next steps:"
echo "1. Prepare your dataset in the data/ directory"
echo "2. Update data/dataset.yaml with your class names"
echo "3. Run: python3 scripts/quick_start.py"
echo
