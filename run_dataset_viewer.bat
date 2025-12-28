@echo off
chcp 65001 >nul
echo Запуск Dataset Viewer...
echo.
echo Использование: python -m video2dataset.cli view-dataset --images data/images --labels data/labels --port 5000
echo.
python -m video2dataset.cli view-dataset %*
pause

