# -*- coding: utf-8 -*-
"""
Модуль для интеграции с обучением YOLOv8
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import yaml
import random
import shutil

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

from .annotation_formats import get_class_distribution
from .config import DEFAULT_ENCODING

logger = logging.getLogger(__name__)


def create_data_yaml(images_dir: Union[str, Path], labels_dir: Union[str, Path],
                    output_path: Union[str, Path], class_names: Optional[List[str]] = None,
                    train_split: float = 0.7, val_split: float = 0.2, test_split: float = 0.1) -> str:
    """
    Создать конфигурационный файл data.yaml для обучения YOLOv8

    Args:
        images_dir: Директория с изображениями
        labels_dir: Директория с аннотациями
        output_path: Путь для сохранения data.yaml
        class_names: Список имен классов (если None, будут сгенерированы автоматически)
        train_split: Доля данных для обучения
        val_split: Доля данных для валидации
        test_split: Доля данных для тестирования

    Returns:
        Путь к созданному файлу data.yaml
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_path = Path(output_path)

    if not images_dir.exists():
        raise FileNotFoundError(f"Директория с изображениями не найдена: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Директория с аннотациями не найдена: {labels_dir}")

    # Найти все изображения
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"**/*{ext}"))
        image_files.extend(images_dir.glob(f"**/*{ext.upper()}"))

    if not image_files:
        raise ValueError(f"Изображения не найдены в директории: {images_dir}")

    # Получить распределение классов
    class_dist = get_class_distribution(labels_dir)

    # Определить имена классов
    if class_names is None:
        max_class_id = max(class_dist.keys()) if class_dist else 0
        class_names = [f"class_{i}" for i in range(max_class_id + 1)]
    else:
        # Проверить соответствие количества классов
        if len(class_names) < len(class_dist):
            logger.warning(f"Указано {len(class_names)} имен классов, но найдено {len(class_dist)} классов в данных")
            # Дополнить имена классов
            for i in range(len(class_names), len(class_dist)):
                class_names.append(f"class_{i}")

    # Перемешать файлы для случайного разделения
    random.seed(42)  # Для воспроизводимости
    random.shuffle(image_files)

    # Разделить на train/val/test
    total_files = len(image_files)
    train_end = int(total_files * train_split)
    val_end = train_end + int(total_files * val_split)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    # Создать директории для разделов
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Функция для создания списка путей к изображениям
    def get_image_paths(file_list):
        return [str(img.relative_to(output_path.parent)) for img in file_list]

    # Создать data.yaml
    data_config = {
        'path': str(output_path.parent.absolute()),
        'train': get_image_paths(train_files),
        'val': get_image_paths(val_files),
        'test': get_image_paths(test_files) if test_files else get_image_paths(val_files),  # Использовать val как test если test пустой
        'nc': len(class_names),  # Количество классов
        'names': class_names  # Имена классов
    }

    # Сохранить YAML
    with open(output_path, 'w', encoding=DEFAULT_ENCODING) as f:
        yaml.safe_dump(data_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Создан файл data.yaml: {output_path}")
    logger.info(f"Классы ({len(class_names)}): {class_names}")
    logger.info(f"Разделение данных: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    return str(output_path)


def create_data_yaml_simple(images_dir: Union[str, Path], labels_dir: Union[str, Path],
                           output_path: Union[str, Path], class_names: Optional[List[str]] = None) -> str:
    """
    Создать простой data.yaml без разделения на train/val/test директории

    Args:
        images_dir: Директория с изображениями
        labels_dir: Директория с аннотациями
        output_path: Путь для сохранения data.yaml
        class_names: Список имен классов

    Returns:
        Путь к созданному файлу data.yaml
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_path = Path(output_path)

    # Получить распределение классов
    class_dist = get_class_distribution(labels_dir)

    # Определить имена классов
    if class_names is None:
        max_class_id = max(class_dist.keys()) if class_dist else 0
        class_names = [f"class_{i}" for i in range(max_class_id + 1)]

    # Создать data.yaml
    data_config = {
        'path': str(images_dir.parent.absolute()),
        'train': str(images_dir.relative_to(images_dir.parent)),
        'val': str(images_dir.relative_to(images_dir.parent)),
        'nc': len(class_names),
        'names': class_names
    }

    # Сохранить YAML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding=DEFAULT_ENCODING) as f:
        yaml.safe_dump(data_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Создан простой файл data.yaml: {output_path}")
    return str(output_path)


def split_dataset_into_folders(images_dir: Union[str, Path], labels_dir: Union[str, Path],
                              output_dir: Union[str, Path],
                              train_split: float = 0.7, val_split: float = 0.2, test_split: float = 0.1,
                              copy_files: bool = True) -> Dict[str, str]:
    """
    Разделить датасет на train/val/test папки

    Args:
        images_dir: Исходная директория с изображениями
        labels_dir: Исходная директория с аннотациями
        output_dir: Выходная директория
        train_split: Доля для обучения
        val_split: Доля для валидации
        test_split: Доля для тестирования
        copy_files: Копировать файлы или создать симлинки

    Returns:
        Словарь с путями к созданным директориям
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

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

    # Перемешать файлы
    random.seed(42)
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

    # Копировать/переместить файлы
    for split_name, files in splits.items():
        images_split_dir = output_dir / split_name / "images"
        labels_split_dir = output_dir / split_name / "labels"

        for img_path in files:
            # Копировать изображение
            dst_img = images_split_dir / img_path.name
            if copy_files:
                shutil.copy2(img_path, dst_img)
            else:
                dst_img.symlink_to(img_path)

            # Копировать аннотацию
            label_path = labels_dir / f"{img_path.stem}.txt"
            dst_label = labels_split_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                if copy_files:
                    shutil.copy2(label_path, dst_label)
                else:
                    dst_label.symlink_to(label_path)
            else:
                # Создать пустой файл аннотаций
                dst_label.touch()

    result_paths = {
        'train_images': str(train_images_dir),
        'train_labels': str(train_labels_dir),
        'val_images': str(val_images_dir),
        'val_labels': str(val_labels_dir),
        'test_images': str(test_images_dir),
        'test_labels': str(test_labels_dir)
    }

    logger.info(f"Датасет разделен на train/val/test:")
    for split_name, files in splits.items():
        logger.info(f"  {split_name}: {len(files)} файлов")

    return result_paths


def generate_yolo_config_from_dataset(images_dir: Union[str, Path], labels_dir: Union[str, Path],
                                    output_yaml: Union[str, Path], class_names: Optional[List[str]] = None) -> str:
    """
    Генерировать полный конфиг YOLO из датасета

    Args:
        images_dir: Директория с изображениями
        labels_dir: Директория с аннотациями
        output_yaml: Путь для сохранения data.yaml
        class_names: Имена классов

    Returns:
        Путь к созданному data.yaml
    """
    # Создать разделение на train/val/test
    dataset_dir = Path(output_yaml).parent
    split_dirs = split_dataset_into_folders(
        images_dir, labels_dir, dataset_dir / "dataset_split"
    )

    # Создать data.yaml
    data_yaml_path = create_data_yaml(
        dataset_dir / "dataset_split" / "train" / "images",
        dataset_dir / "dataset_split" / "train" / "labels",
        output_yaml,
        class_names,
        train_split=1.0,  # Все train файлы уже разделены
        val_split=0.0,
        test_split=0.0
    )

    # Обновить пути в data.yaml для корректного указания на разделенные директории
    with open(output_yaml, 'r', encoding=DEFAULT_ENCODING) as f:
        data_config = yaml.safe_load(f)

    # Обновить пути
    base_path = str(dataset_dir.absolute())
    data_config['path'] = base_path
    data_config['train'] = 'dataset_split/train/images'
    data_config['val'] = 'dataset_split/val/images'
    data_config['test'] = 'dataset_split/test/images'

    # Сохранить обновленный конфиг
    with open(output_yaml, 'w', encoding=DEFAULT_ENCODING) as f:
        yaml.safe_dump(data_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Создан полный конфиг YOLO: {output_yaml}")
    return str(output_yaml)


def get_training_suggestion(dataset_stats: Dict, model_size: str = 'n') -> Dict:
    """
    Получить рекомендации по параметрам обучения на основе датасета

    Args:
        dataset_stats: Статистика датасета
        model_size: Размер модели ('n', 's', 'm', 'l', 'x')

    Returns:
        Словарь с рекомендованными параметрами
    """
    # Рекомендации по размеру изображения
    img_size = 640
    if dataset_stats.get('avg_image_size', 640) < 400:
        img_size = 416
    elif dataset_stats.get('avg_image_size', 640) > 1000:
        img_size = 1280

    # Рекомендации по количеству эпох
    num_classes = dataset_stats.get('num_classes', 1)
    total_samples = dataset_stats.get('total_samples', 1000)

    if total_samples < 1000:
        epochs = 100
    elif total_samples < 5000:
        epochs = 50
    else:
        epochs = 30

    # Batch size в зависимости от размера модели
    batch_sizes = {'n': 16, 's': 8, 'm': 4, 'l': 2, 'x': 1}
    batch_size = batch_sizes.get(model_size.lower(), 8)

    # Рекомендации по learning rate
    if num_classes <= 10:
        lr0 = 0.01
    else:
        lr0 = 0.001

    return {
        'model': f'yolov8{model_size.lower()}.pt',
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'lr0': lr0,
        'optimizer': 'AdamW',
        'augment': True,
        'patience': 20
    }


def create_training_script(data_yaml: Union[str, Path], output_script: Union[str, Path],
                          training_params: Optional[Dict] = None) -> str:
    """
    Создать скрипт для обучения YOLO

    Args:
        data_yaml: Путь к data.yaml
        output_script: Путь для сохранения скрипта обучения
        training_params: Параметры обучения

    Returns:
        Путь к созданному скрипту
    """
    if training_params is None:
        training_params = {
            'model': 'yolov8n.pt',
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
            'lr0': 0.01
        }

    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обучения YOLOv8 на подготовленном датасете
"""

from ultralytics import YOLO

def main():
    # Загрузка модели
    model = YOLO('{training_params["model"]}')

    # Обучение
    results = model.train(
        data='{str(data_yaml)}',
        epochs={training_params["epochs"]},
        imgsz={training_params["imgsz"]},
        batch={training_params["batch"]},
        lr0={training_params["lr0"]},
        optimizer='AdamW',
        augment=True,
        patience=20,
        save=True,
        project='yolo_training',
        name='experiment_1'
    )

    print("Обучение завершено!")
    print(f"Результаты сохранены в: {{results.save_dir}}")

if __name__ == "__main__":
    main()
'''

    output_script = Path(output_script)
    output_script.parent.mkdir(parents=True, exist_ok=True)

    with open(output_script, 'w', encoding=DEFAULT_ENCODING) as f:
        f.write(script_content)

    # Сделать исполняемым на Unix системах
    try:
        import os
        os.chmod(output_script, 0o755)
    except:
        pass

    logger.info(f"Создан скрипт обучения: {output_script}")
    return str(output_script)
