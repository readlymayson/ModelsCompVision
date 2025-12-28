# -*- coding: utf-8 -*-
"""
Глобальные настройки проекта Video2Dataset
"""

import os
from pathlib import Path

# Пути к директориям по умолчанию
DEFAULT_DATA_DIR = Path("data")
DEFAULT_VIDEOS_DIR = DEFAULT_DATA_DIR / "videos"
DEFAULT_IMAGES_DIR = DEFAULT_DATA_DIR / "images"
DEFAULT_LABELS_DIR = DEFAULT_DATA_DIR / "labels"
DEFAULT_AUGMENTED_DIR = DEFAULT_DATA_DIR / "augmented"
DEFAULT_PREVIEWS_DIR = DEFAULT_DATA_DIR / "previews"

# Настройки разбиения видео
DEFAULT_VIDEO_FPS = 1.0  # кадры в секунду (по умолчанию 1 кадр в секунду)
DEFAULT_VIDEO_EVERY_N = None  # каждый N-й кадр (None = использовать fps)
DEFAULT_IMAGE_FORMAT = "jpg"
DEFAULT_IMAGE_QUALITY = 95

# Настройки YOLO
DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

# Настройки аугментаций
DEFAULT_NUM_AUGS_PER_IMAGE = 3
DEFAULT_AUGMENTATION_PROBABILITY = 0.8

# Кодировка файлов
DEFAULT_ENCODING = "utf-8"

def get_project_root():
    """Получить корневую директорию проекта"""
    return Path(__file__).parent.parent

def ensure_directories_exist(*dirs):
    """Убедиться, что директории существуют"""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def setup_default_directories():
    """Создать директории по умолчанию, если они не существуют"""
    ensure_directories_exist(
        DEFAULT_VIDEOS_DIR,
        DEFAULT_IMAGES_DIR,
        DEFAULT_LABELS_DIR,
        DEFAULT_AUGMENTED_DIR,
        DEFAULT_PREVIEWS_DIR
    )
