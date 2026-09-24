#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting up DeepGuard prototype..."

# Check if python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install Python3."
    exit 1
fi

echo "Creating virtual environment 'venv'..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete! Run 'source venv/bin/activate' to activate the environment."
echo "Then, run 'python check_install.py' to verify the installation."
