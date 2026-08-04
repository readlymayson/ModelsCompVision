#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки пагинации API
"""

import sys
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from video2dataset.dataset_viewer import create_dataset_viewer_app

def test_pagination():
    """Тестирование пагинации API"""
    images_dir = Path("data/images")
    labels_dir = Path("data/labels")

    if not images_dir.exists() or not labels_dir.exists():
        print("Директории data/images или data/labels не найдены")
        return

    # Создаем тестовое приложение
    app = create_dataset_viewer_app(images_dir, labels_dir)

    # Тестируем с тестовым клиентом Flask
    with app.test_client() as client:
        print("=== Тест пагинации API ===")

        # Проверим первую страницу
        response1 = client.get('/api/images?page=1&per_page=5')
        data1 = response1.get_json()
        images1 = [img['name'] for img in data1['images']]
        print(f"Страница 1: {len(images1)} изображений")
        print(f"Имена: {images1}")

        # Проверим вторую страницу
        response2 = client.get('/api/images?page=2&per_page=5')
        data2 = response2.get_json()
        images2 = [img['name'] for img in data2['images']]
        print(f"Страница 2: {len(images2)} изображений")
        print(f"Имена: {images2}")

        # Проверим на дубликаты между страницами
        duplicates = set(images1) & set(images2)
        if duplicates:
            print(f"❌ Найдены дубликаты между страницами: {duplicates}")
        else:
            print("✅ Дубликатов между страницами нет")

        # Проверим общее количество
        total = data1['total']
        print(f"Общее количество изображений: {total}")

        # Проверим, что страницы не пересекаются
        all_images = set(images1 + images2)
        expected_count = min(10, total)  # 2 страницы по 5 изображений
        if len(all_images) == expected_count:
            print("✅ Страницы не пересекаются")
        else:
            print(f"❌ Ожидалось {expected_count} уникальных изображений, получено {len(all_images)}")

if __name__ == "__main__":
    test_pagination()
