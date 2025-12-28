# -*- coding: utf-8 -*-
"""
Скрипт для фильтрации датасета с пустыми аннотациями.
Перемещает изображения и соответствующие аннотационные файлы без аннотаций
в отдельную папку "неаннотированно".
"""

import os
import shutil
from pathlib import Path


def is_annotation_file_empty(annotation_file_path):
    """
    Проверяет, пуст ли файл аннотаций.

    Args:
        annotation_file_path (str): Путь к файлу аннотаций

    Returns:
        bool: True если файл пустой или содержит только пустые строки
    """
    try:
        with open(annotation_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Файл считается пустым если он пустой или содержит только пустые строки
            return len(content) == 0
    except FileNotFoundError:
        print(f"Файл аннотаций не найден: {annotation_file_path}")
        return False
    except Exception as e:
        print(f"Ошибка при чтении файла {annotation_file_path}: {e}")
        return False


def get_image_path_from_annotation(annotation_path, images_dir):
    """
    Получает путь к изображению на основе пути к файлу аннотаций.

    Args:
        annotation_path (str): Путь к файлу аннотаций
        images_dir (str): Путь к папке с изображениями

    Returns:
        str: Путь к изображению или None если не найден
    """
    # Получаем имя файла аннотаций без расширения
    annotation_name = Path(annotation_path).stem

    # Ищем соответствующее изображение
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_path = os.path.join(images_dir, annotation_name + ext)
        if os.path.exists(image_path):
            return image_path

    return None


def create_unannotated_folders(base_dir):
    """
    Создает структуру папок для неаннотированных данных.

    Args:
        base_dir (str): Базовая директория для создания папок
    """
    unannotated_dir = os.path.join(base_dir, "неаннотированно")
    images_dir = os.path.join(unannotated_dir, "images")
    labels_dir = os.path.join(unannotated_dir, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    return unannotated_dir, images_dir, labels_dir


def filter_empty_annotations(data_dir="data"):
    """
    Фильтрует датасет, перемещая пустые аннотации в отдельную папку.

    Args:
        data_dir (str): Путь к папке с данными (по умолчанию "data")
    """
    labels_dir = os.path.join(data_dir, "labels")
    images_dir = os.path.join(data_dir, "images")

    if not os.path.exists(labels_dir):
        print(f"Папка с аннотациями не найдена: {labels_dir}")
        return

    if not os.path.exists(images_dir):
        print(f"Папка с изображениями не найдена: {images_dir}")
        return

    # Создаем папки для неаннотированных данных
    unannotated_dir, unannotated_images_dir, unannotated_labels_dir = create_unannotated_folders(data_dir)

    print(f"Начинаем фильтрацию датасета в папке: {data_dir}")
    print(f"Неаннотированные файлы будут перемещены в: {unannotated_dir}")

    # Счетчики для статистики
    total_files = 0
    empty_files = 0

    # Проходим по всем файлам аннотаций
    for filename in os.listdir(labels_dir):
        if not filename.endswith('.txt'):
            continue

        total_files += 1
        annotation_path = os.path.join(labels_dir, filename)

        # Проверяем, пуст ли файл аннотаций
        if is_annotation_file_empty(annotation_path):
            empty_files += 1

            # Находим соответствующее изображение
            image_path = get_image_path_from_annotation(annotation_path, images_dir)

            if image_path:
                # Пути назначения
                dest_image_path = os.path.join(unannotated_images_dir, os.path.basename(image_path))
                dest_annotation_path = os.path.join(unannotated_labels_dir, filename)

                try:
                    # Перемещаем изображение
                    shutil.move(image_path, dest_image_path)
                    print(f"Перемещено изображение: {os.path.basename(image_path)}")

                    # Перемещаем файл аннотаций
                    shutil.move(annotation_path, dest_annotation_path)
                    print(f"Перемещен файл аннотаций: {filename}")

                except Exception as e:
                    print(f"Ошибка при перемещении файлов {filename}: {e}")
            else:
                print(f"Изображение не найдено для аннотационного файла: {filename}")

    print("\nФильтрация завершена!")
    print(f"Всего обработано файлов: {total_files}")
    print(f"Перемещено пустых аннотаций: {empty_files}")
    print(f"Осталось аннотированных файлов: {total_files - empty_files}")


if __name__ == "__main__":
    # Запуск фильтрации
    filter_empty_annotations()
