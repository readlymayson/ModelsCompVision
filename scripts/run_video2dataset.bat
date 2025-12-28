@echo off
chcp 65001 >nul
python -m video2dataset.cli %*
if errorlevel 1 (
    echo.
    echo Для получения справки выполните: run_video2dataset.bat --help
    pause
)
