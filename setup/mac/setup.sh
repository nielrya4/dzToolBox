#!/bin/bash
cd ../..
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed."
    if ! command -v brew &> /dev/null; then
        echo "Homebrew is not installed. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    echo "Installing Python3 via Homebrew..."
    brew install python3
else
    echo "Python3 is already installed."
fi
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating venv..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi
. venv/bin/activate
pip install -r requirements.txt
