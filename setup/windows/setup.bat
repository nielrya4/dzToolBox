@echo off
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python 3 is not installed.
    where choco >nul 2>nul
    if %errorlevel% neq 0 (
        echo Chocolatey is not installed. Installing Chocolatey...
        @powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    ) else (
        echo Chocolatey is already installed.
    )
    choco install python -y
) else (
    echo Python 3 is already installed.
)
cd ..\..
if not exist venv (
    echo Virtual environment not found. Creating venv...
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)
call venv\Scripts\activate
pip install -r requirements.txt
