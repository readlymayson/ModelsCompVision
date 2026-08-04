@echo off
chcp 65001 >nul
echo Активация venv и проверка CUDA...
cd /d E:\CodeProjects\ModelsCompVision
call venv\Scripts\activate.bat
python check_cuda.py
pause





