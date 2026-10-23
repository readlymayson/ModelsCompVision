@echo off
chcp 65001 >nul
echo Активация venv и проверка CUDA...
cd /d %~dp0
call venv\Scripts\activate.bat
python check_cuda.py
pause





