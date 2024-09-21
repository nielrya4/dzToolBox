@echo off
cd ..\..
netsh advfirewall firewall show rule name="Allow Port 5000" >nul 2>nul
if %errorlevel% neq 0 (
    echo Allowing traffic on port 5000...
    netsh advfirewall firewall add rule name="Allow Port 5000" dir=in action=allow protocol=TCP localport=5000
) else (
    echo Port 5000 is already allowed.
)
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr "IPv4"') do (
    set "IP=%%i"
    setlocal enabledelayedexpansion
    set "IP=!IP: =!"
    endlocal
)
start http://%IP%:5000
uwsgi --socket 0.0.0.0:5000 --protocol=http -w wsgi:app
