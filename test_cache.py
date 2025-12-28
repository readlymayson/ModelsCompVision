#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки кэширования
"""

import sys
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from video2dataset.dataset_viewer import create_dataset_viewer_app

def test_cache_sync():
    """Тестирование синхронизации кэшей"""
    images_dir = Path("data/images")
    labels_dir = Path("data/labels")

    if not images_dir.exists() or not labels_dir.exists():
        print("Директории data/images или data/labels не найдены")
        return

    # Создаем тестовое приложение
    app = create_dataset_viewer_app(images_dir, labels_dir)

    # Тестируем с тестовым клиентом Flask
    with app.test_client() as client:
        print("=== Тест синхронизации кэшей ===")

        # Сначала проверяем count
        print("1. Запрос count...")
        count_response = client.get('/api/images/count')
        count_data = count_response.get_json()
        print(f"   Count: {count_data.get('count', 'error')}")

        # Проверяем, создался ли кэш
        print(f"   Image cache exists: {hasattr(app, 'image_cache')}")
        if hasattr(app, 'image_cache'):
            print(f"   Image cache keys: {list(app.image_cache.keys())}")

        # Теперь запрашиваем images
        print("2. Запрос images...")
        images_response = client.get('/api/images?page=1&per_page=5')
        images_data = images_response.get_json()
        print(f"   Images count: {len(images_data.get('images', []))}")
        print(f"   Total: {images_data.get('total', 'error')}")

        # Проверяем, что count и total совпадают
        count_from_api = count_data.get('count', 0)
        total_from_api = images_data.get('total', 0)

        if count_from_api == total_from_api:
            print("✅ Кэши синхронизированы!")
        else:
            print(f"❌ Несоответствие: count={count_from_api}, total={total_from_api}")

if __name__ == "__main__":
    test_cache_sync()

