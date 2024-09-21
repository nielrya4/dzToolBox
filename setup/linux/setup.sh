#!/bin/bash
cd ../..
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed."
    if [ -f /etc/debian_version ]; then
        echo "Installing Python3 on a Debian-based system..."
        sudo apt update
        sudo apt install python3 python3-venv python3-pip -y
    elif [ -f /etc/redhat-release ]; then
        echo "Installing Python3 on a RedHat-based system..."
        sudo yum install python3 -y
    elif [ -f /etc/fedora-release ]; then
        echo "Installing Python3 on a Fedora system..."
        sudo dnf install python3 -y
    else
        echo "Unsupported Linux distribution. Please install Python3 manually."
        exit 1
    fi
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
pwd
. venv/bin/activate
pip install -r requirements.txt
