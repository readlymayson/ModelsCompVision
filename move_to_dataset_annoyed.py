# -*- coding: utf-8 -*-
"""
Скрипт для переноса файлов из data\images и data\labels в dataset_annoyed.
Перемещает изображения и соответствующие аннотационные файлы в dataset_annoyed.
"""

import os
import shutil
from pathlib import Path


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


def get_annotation_path_from_image(image_path, labels_dir):
    """
    Получает путь к файлу аннотаций на основе пути к изображению.

    Args:
        image_path (str): Путь к изображению
        labels_dir (str): Путь к папке с аннотациями

    Returns:
        str: Путь к файлу аннотаций или None если не найден
    """
    # Получаем имя файла изображения без расширения
    image_name = Path(image_path).stem
    annotation_path = os.path.join(labels_dir, image_name + '.txt')
    
    if os.path.exists(annotation_path):
        return annotation_path
    
    return None


def move_to_dataset_annoyed(data_dir="data", target_dir="dataset_annoyed"):
    """
    Переносит файлы из data в dataset_annoyed.

    Args:
        data_dir (str): Путь к исходной папке с данными (по умолчанию "data")
        target_dir (str): Путь к целевой папке (по умолчанию "dataset_annoyed")
    """
    source_images_dir = os.path.join(data_dir, "images")
    source_labels_dir = os.path.join(data_dir, "labels")
    
    target_images_dir = os.path.join(target_dir, "images")
    target_labels_dir = os.path.join(target_dir, "labels")

    if not os.path.exists(source_images_dir):
        print(f"Папка с изображениями не найдена: {source_images_dir}")
        return

    if not os.path.exists(source_labels_dir):
        print(f"Папка с аннотациями не найдена: {source_labels_dir}")
        return

    # Создаем целевые папки если их нет
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)

    print(f"Начинаем перенос файлов из {data_dir} в {target_dir}")
    print(f"Изображения: {source_images_dir} -> {target_images_dir}")
    print(f"Аннотации: {source_labels_dir} -> {target_labels_dir}")

    # Счетчики для статистики
    images_moved = 0
    labels_moved = 0
    images_without_labels = 0
    labels_without_images = 0
    errors = 0

    # Сначала обрабатываем изображения и находим соответствующие аннотации
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    processed_images = set()
    processed_labels = set()

    # Проходим по всем изображениям
    for filename in os.listdir(source_images_dir):
        file_path = os.path.join(source_images_dir, filename)
        
        if not os.path.isfile(file_path):
            continue
        
        # Проверяем расширение
        if not any(filename.lower().endswith(ext) for ext in image_extensions):
            continue

        image_name = Path(filename).stem
        processed_images.add(image_name)

        # Находим соответствующую аннотацию
        annotation_path = get_annotation_path_from_image(file_path, source_labels_dir)
        
        try:
            # Перемещаем изображение
            dest_image_path = os.path.join(target_images_dir, filename)
            shutil.move(file_path, dest_image_path)
            images_moved += 1
            print(f"Перемещено изображение: {filename}")

            # Перемещаем аннотацию если она существует
            if annotation_path:
                annotation_filename = os.path.basename(annotation_path)
                dest_annotation_path = os.path.join(target_labels_dir, annotation_filename)
                shutil.move(annotation_path, dest_annotation_path)
                labels_moved += 1
                processed_labels.add(image_name)
                print(f"Перемещен файл аннотаций: {annotation_filename}")
            else:
                images_without_labels += 1
                print(f"Аннотация не найдена для изображения: {filename}")

        except Exception as e:
            errors += 1
            print(f"Ошибка при перемещении файла {filename}: {e}")

    # Обрабатываем оставшиеся аннотации без соответствующих изображений
    for filename in os.listdir(source_labels_dir):
        if not filename.endswith('.txt'):
            continue

        annotation_name = Path(filename).stem
        
        # Пропускаем уже обработанные аннотации
        if annotation_name in processed_labels:
            continue

        annotation_path = os.path.join(source_labels_dir, filename)
        
        try:
            dest_annotation_path = os.path.join(target_labels_dir, filename)
            shutil.move(annotation_path, dest_annotation_path)
            labels_moved += 1
            labels_without_images += 1
            print(f"Перемещен файл аннотаций без изображения: {filename}")
        except Exception as e:
            errors += 1
            print(f"Ошибка при перемещении файла аннотаций {filename}: {e}")

    print("\nПеренос завершен!")
    print(f"Перемещено изображений: {images_moved}")
    print(f"Перемещено аннотаций: {labels_moved}")
    print(f"Изображений без аннотаций: {images_without_labels}")
    print(f"Аннотаций без изображений: {labels_without_images}")
    print(f"Ошибок: {errors}")


if __name__ == "__main__":
    # Запуск переноса
    move_to_dataset_annoyed()








