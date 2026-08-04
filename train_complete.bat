@echo off
chcp 65001 >nul

echo === Подготовка датасета ===
python prepare_dataset.py

echo.
echo === Запуск обучения YOLO модели ===
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); results = model.train(data='data.yaml', epochs=10, imgsz=420, project='runs', name='yolo_training')"

echo.
echo === Обучение завершено! ===
echo Результаты сохранены в папке runs\yolo_training\





