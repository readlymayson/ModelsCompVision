#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки разбиения видео
"""

import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from video2dataset.video_splitter import split_video
    from pathlib import Path

    print("Тестирую разбиение видео...")

    # Проверяем существование видео
    video_path = Path("data/videos/2025-12-25.mp4")
    output_dir = Path("data/images")

    if not video_path.exists():
        print(f"Видео не найдено: {video_path}")
        sys.exit(1)

    print(f"Видео найдено: {video_path}")

    # Запускаем разбиение
    saved_paths = split_video(
        video_path=str(video_path),
        output_dir=str(output_dir),
        fps=1.0  # 1 кадр в секунду
    )

    print(f"Извлечено кадров: {len(saved_paths)}")
    for path in saved_paths[:5]:  # Покажем первые 5
        print(f"  {path}")

    if len(saved_paths) > 5:
        print(f"  ... и еще {len(saved_paths) - 5} файлов")

except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Установите зависимости: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Ошибка: {e}")
    sys.exit(1)
