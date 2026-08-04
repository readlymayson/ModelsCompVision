# -*- coding: utf-8 -*-
"""
Скрипт для подготовки датасета: разделение на train/val/test и создание data.yaml
"""

import os
import random
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

# Импорты из video2dataset (прямой импорт для избежания загрузки Flask)
from pathlib import Path
import yaml

DEFAULT_ENCODING = 'utf-8'

def get_class_distribution(labels_dir: Union[str, Path]) -> Dict[int, int]:
    """
    Получить распределение классов в датасете YOLO

    Args:
        labels_dir: Директория с аннотациями

    Returns:
        Словарь {class_id: count}
    """
    labels_dir = Path(labels_dir)
    class_dist = {}

    if not labels_dir.exists():
        return class_dist

    # Найти все .txt файлы
    for txt_file in labels_dir.glob('*.txt'):
        try:
            with open(txt_file, 'r', encoding=DEFAULT_ENCODING) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 1:
                            class_id = int(parts[0])
                            class_dist[class_id] = class_dist.get(class_id, 0) + 1
        except Exception as e:
            print(f"Ошибка чтения {txt_file}: {e}")
            continue

    return class_dist


def split_dataset(images_dir: Union[str, Path], labels_dir: Union[str, Path],
                  output_dir: Union[str, Path],
                  train_split: float = 0.7, val_split: float = 0.2, test_split: float = 0.1,
                  seed: int = 42) -> Dict[str, str]:
    """
    Разделить датасет на train/val/test папки

    Args:
        images_dir: Исходная директория с изображениями
        labels_dir: Исходная директория с аннотациями
        output_dir: Выходная директория
        train_split: Доля для обучения (0.0-1.0)
        val_split: Доля для валидации (0.0-1.0)
        test_split: Доля для тестирования (0.0-1.0)
        seed: Seed для воспроизводимости

    Returns:
        Словарь с путями к созданным директориям
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Директория с изображениями не найдена: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Директория с аннотациями не найдена: {labels_dir}")

    # Создать структуру директорий
    train_images_dir = output_dir / "train" / "images"
    train_labels_dir = output_dir / "train" / "labels"
    val_images_dir = output_dir / "val" / "images"
    val_labels_dir = output_dir / "val" / "labels"
    test_images_dir = output_dir / "test" / "images"
    test_labels_dir = output_dir / "test" / "labels"

    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir,
                     test_images_dir, test_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Найти все изображения
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"**/*{ext}"))
        image_files.extend(images_dir.glob(f"**/*{ext.upper()}"))

    if not image_files:
        raise FileNotFoundError(f"Изображения не найдены в директории: {images_dir}")

    # Перемешать файлы
    random.seed(seed)
    random.shuffle(image_files)

    # Разделить
    total_files = len(image_files)
    train_end = int(total_files * train_split)
    val_end = train_end + int(total_files * val_split)

    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    # Копировать файлы
    stats = {'train': 0, 'val': 0, 'test': 0}

    for split_name, files in splits.items():
        images_split_dir = output_dir / split_name / "images"
        labels_split_dir = output_dir / split_name / "labels"

        for img_path in files:
            # Копировать изображение
            dst_img = images_split_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            # Копировать аннотацию
            label_path = labels_dir / f"{img_path.stem}.txt"
            dst_label = labels_split_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, dst_label)
            else:
                # Создать пустой файл аннотаций
                dst_label.touch()

            stats[split_name] += 1

    result_paths = {
        'train_images': str(train_images_dir),
        'train_labels': str(train_labels_dir),
        'val_images': str(val_images_dir),
        'val_labels': str(val_labels_dir),
        'test_images': str(test_images_dir),
        'test_labels': str(test_labels_dir)
    }

    print("Датасет разделен:")
    for split_name, count in stats.items():
        print(f"  {split_name}: {count} файлов")

    return result_paths


def create_data_yaml(train_images_dir: Union[str, Path], val_images_dir: Union[str, Path],
                    test_images_dir: Union[str, Path], train_labels_dir: Union[str, Path],
                    output_path: Union[str, Path], class_names: Optional[List[str]] = None) -> str:
    """
    Создать data.yaml файл для YOLO обучения

    Args:
        train_images_dir: Директория с train изображениями
        val_images_dir: Директория с val изображениями
        test_images_dir: Директория с test изображениями
        train_labels_dir: Директория с train аннотациями (для определения классов)
        output_path: Путь для сохранения data.yaml
        class_names: Список имен классов (если None, генерируются автоматически)

    Returns:
        Путь к созданному файлу data.yaml
    """
    train_images_dir = Path(train_images_dir)
    train_labels_dir = Path(train_labels_dir)
    output_path = Path(output_path)

    # Получить распределение классов из train аннотаций
    class_dist = get_class_distribution(train_labels_dir)

    # Определить имена классов
    if class_names is None:
        max_class_id = max(class_dist.keys()) if class_dist else 0
        class_names = [f"class_{i}" for i in range(max_class_id + 1)]

    # Создать data.yaml
    data_config = {
        'path': str(train_images_dir.parent.parent.absolute()),
        'train': str(train_images_dir.relative_to(train_images_dir.parent.parent)),
        'val': str(Path(val_images_dir).relative_to(Path(val_images_dir).parent.parent)),
        'test': str(Path(test_images_dir).relative_to(Path(test_images_dir).parent.parent)),
        'nc': len(class_names),
        'names': class_names
    }

    # Сохранить YAML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding=DEFAULT_ENCODING) as f:
        yaml.safe_dump(data_config, f, default_flow_style=False, sort_keys=False)

    print(f"Создан data.yaml: {output_path}")
    print(f"Классы: {class_names}")
    print(f"Распределение классов в train: {class_dist}")

    return str(output_path)


def prepare_dataset_for_training(images_dir="dataset_annoyed/images",
                                labels_dir="dataset_annoyed/labels",
                                output_dir="dataset_split",
                                data_yaml_path="data.yaml",
                                train_split=0.7, val_split=0.2, test_split=0.1,
                                class_names=None,
                                force_recreate=False):
    """
    Полная подготовка датасета для обучения YOLO

    Args:
        images_dir: Директория с изображениями
        labels_dir: Директория с аннотациями
        output_dir: Выходная директория для разделенного датасета
        data_yaml_path: Путь для сохранения data.yaml
        train_split: Доля для обучения
        val_split: Доля для валидации
        test_split: Доля для тестирования
        class_names: Список имен классов
        force_recreate: Принудительно пересоздать датасет если существует
    """
    output_dir_path = Path(output_dir)

    # Проверить существует ли уже разделенный датасет
    if output_dir_path.exists() and not force_recreate:
        train_dir = output_dir_path / "train"
        if train_dir.exists() and any(train_dir.rglob("*")):
            print(f"Датасет уже существует в {output_dir}")
            print("Используем существующий датасет")
            print("Используйте force_recreate=True для пересоздания")

            # Создать data.yaml для существующего датасета
            paths = {
                'train_images': str(output_dir_path / "train" / "images"),
                'val_images': str(output_dir_path / "val" / "images"),
                'test_images': str(output_dir_path / "test" / "images"),
                'train_labels': str(output_dir_path / "train" / "labels")
            }

            data_yaml = create_data_yaml(
                paths['train_images'],
                paths['val_images'],
                paths['test_images'],
                paths['train_labels'],
                data_yaml_path,
                class_names
            )

            print(f"\nData YAML обновлен: {data_yaml}")
            return

    print("Начинаем подготовку датасета...")
    print(f"Исходные данные: {images_dir}, {labels_dir}")
    print(f"Выходная директория: {output_dir}")

    # Удалить старую директорию если force_recreate
    if output_dir_path.exists() and force_recreate:
        print(f"Удаляем старую директорию {output_dir}")
        shutil.rmtree(output_dir_path)

    # Разделить датасет
    paths = split_dataset(images_dir, labels_dir, output_dir,
                         train_split, val_split, test_split)

    # Создать data.yaml
    data_yaml = create_data_yaml(
        paths['train_images'],
        paths['val_images'],
        paths['test_images'],
        paths['train_labels'],
        data_yaml_path,
        class_names
    )

    print("\nПодготовка завершена!")
    print(f"Data YAML создан: {data_yaml}")
    print("Можно запускать обучение YOLO!")


def main():
    """Главная функция с парсером аргументов"""
    parser = argparse.ArgumentParser(
        description="Подготовка датасета для обучения YOLO: разделение на train/val/test и создание data.yaml"
    )

    parser.add_argument('--images-dir', default='dataset_annoyed/images',
                       help='Директория с изображениями (по умолчанию: dataset_annoyed/images)')
    parser.add_argument('--labels-dir', default='dataset_annoyed/labels',
                       help='Директория с аннотациями (по умолчанию: dataset_annoyed/labels)')
    parser.add_argument('--output-dir', default='dataset_split',
                       help='Выходная директория для разделенного датасета (по умолчанию: dataset_split)')
    parser.add_argument('--data-yaml', default='data.yaml',
                       help='Путь для сохранения data.yaml (по умолчанию: data.yaml)')
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='Доля для обучения (0.0-1.0, по умолчанию: 0.7)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Доля для валидации (0.0-1.0, по умолчанию: 0.2)')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Доля для тестирования (0.0-1.0, по умолчанию: 0.1)')
    parser.add_argument('--force-recreate', action='store_true',
                       help='Принудительно пересоздать датасет если существует')
    parser.add_argument('--class-names', nargs='+',
                       help='Имена классов (если не указаны, генерируются автоматически)')

    args = parser.parse_args()

    # Проверка сумм долей
    total_split = args.train_split + args.val_split + args.test_split
    if abs(total_split - 1.0) > 1e-6:
        parser.error(f"Сумма долей должна быть равна 1.0, текущая сумма: {total_split}")

    # Запуск подготовки датасета
    prepare_dataset_for_training(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        data_yaml_path=args.data_yaml,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        class_names=args.class_names,
        force_recreate=args.force_recreate
    )


if __name__ == "__main__":
    main()
