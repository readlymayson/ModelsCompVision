# -*- coding: utf-8 -*-
"""
Модуль для работы с аннотациями в формате YOLOv8
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import cv2
import numpy as np

from .config import DEFAULT_ENCODING

logger = logging.getLogger(__name__)


class YOLOAnnotation:
    """Класс для работы с аннотацией YOLO для одного объекта"""

    def __init__(self, class_id: int, x_center: float, y_center: float,
                 width: float, height: float):
        """
        Инициализация аннотации YOLO

        Args:
            class_id: ID класса объекта
            x_center: X координата центра (нормализованная, 0-1)
            y_center: Y координата центра (нормализованная, 0-1)
            width: Ширина bounding box (нормализованная, 0-1)
            height: Высота bounding box (нормализованная, 0-1)
        """
        self.class_id = int(class_id)
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.width = float(width)
        self.height = float(height)

    def to_pixels(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Преобразовать нормализованные координаты в пиксели

        Args:
            img_width: Ширина изображения
            img_height: Высота изображения

        Returns:
            Кортеж (x1, y1, x2, y2) - координаты bounding box в пикселях
        """
        x1 = int((self.x_center - self.width / 2) * img_width)
        y1 = int((self.y_center - self.height / 2) * img_height)
        x2 = int((self.x_center + self.width / 2) * img_width)
        y2 = int((self.y_center + self.height / 2) * img_height)

        return x1, y1, x2, y2

    def to_coco_bbox(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Преобразовать в формат COCO bbox [x, y, width, height]

        Args:
            img_width: Ширина изображения
            img_height: Высота изображения

        Returns:
            Кортеж (x, y, width, height) в пикселях
        """
        x1, y1, x2, y2 = self.to_pixels(img_width, img_height)
        return x1, y1, x2 - x1, y2 - y1

    @classmethod
    def from_pixels(cls, class_id: int, x1: int, y1: int, x2: int, y2: int,
                   img_width: int, img_height: int) -> 'YOLOAnnotation':
        """
        Создать аннотацию YOLO из координат в пикселях

        Args:
            class_id: ID класса
            x1, y1, x2, y2: Координаты bounding box в пикселях
            img_width: Ширина изображения
            img_height: Высота изображения

        Returns:
            Экземпляр YOLOAnnotation
        """
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        return cls(class_id, x_center, y_center, width, height)

    @classmethod
    def from_coco_bbox(cls, class_id: int, x: int, y: int, width: int, height: int,
                      img_width: int, img_height: int) -> 'YOLOAnnotation':
        """
        Создать аннотацию YOLO из COCO bbox формата

        Args:
            class_id: ID класса
            x, y: Координаты верхнего левого угла
            width, height: Размеры bounding box
            img_width: Ширина изображения
            img_height: Высота изображения

        Returns:
            Экземпляр YOLOAnnotation
        """
        x1, y1 = x, y
        x2, y2 = x + width, y + height
        return cls.from_pixels(class_id, x1, y1, x2, y2, img_width, img_height)

    def validate(self) -> bool:
        """
        Проверить корректность аннотации

        Returns:
            True если аннотация корректна
        """
        # Проверка диапазонов координат
        coords = [self.x_center, self.y_center, self.width, self.height]
        if not all(0 <= coord <= 1 for coord in coords):
            return False

        # Проверка что размеры положительные
        if self.width <= 0 or self.height <= 0:
            return False

        # Проверка что bounding box не выходит за границы
        if (self.x_center - self.width/2 < 0 or
            self.x_center + self.width/2 > 1 or
            self.y_center - self.height/2 < 0 or
            self.y_center + self.height/2 > 1):
            return False

        return True

    def clamp_to_bounds(self) -> 'YOLOAnnotation':
        """
        Привести координаты к допустимым границам

        Returns:
            Исправленная аннотация
        """
        # Ограничить координаты диапазоном [0, 1]
        self.x_center = max(0, min(1, self.x_center))
        self.y_center = max(0, min(1, self.y_center))
        self.width = max(0, min(1, self.width))
        self.height = max(0, min(1, self.height))

        # Убедиться что bbox не выходит за границы
        x1 = self.x_center - self.width / 2
        x2 = self.x_center + self.width / 2
        y1 = self.y_center - self.height / 2
        y2 = self.y_center + self.height / 2

        if x1 < 0:
            self.x_center += (0 - x1)
        if x2 > 1:
            self.x_center -= (x2 - 1)
        if y1 < 0:
            self.y_center += (0 - y1)
        if y2 > 1:
            self.y_center -= (y2 - 1)

        return self

    def __str__(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def __repr__(self) -> str:
        return f"YOLOAnnotation(class_id={self.class_id}, center=({self.x_center:.3f}, {self.y_center:.3f}), size=({self.width:.3f}, {self.height:.3f}))"


class YOLOAnnotations:
    """Класс для работы с набором аннотаций YOLO для одного изображения"""

    def __init__(self, annotations: Optional[List[YOLOAnnotation]] = None):
        """
        Инициализация набора аннотаций

        Args:
            annotations: Список аннотаций YOLOAnnotation
        """
        self.annotations = annotations or []

    def add_annotation(self, annotation: YOLOAnnotation):
        """Добавить аннотацию"""
        self.annotations.append(annotation)

    def remove_annotation(self, index: int):
        """Удалить аннотацию по индексу"""
        if 0 <= index < len(self.annotations):
            del self.annotations[index]

    def get_annotations_by_class(self, class_id: int) -> List[YOLOAnnotation]:
        """Получить все аннотации определенного класса"""
        return [ann for ann in self.annotations if ann.class_id == class_id]

    def validate_all(self) -> List[str]:
        """
        Проверить все аннотации

        Returns:
            Список ошибок валидации
        """
        errors = []
        for i, ann in enumerate(self.annotations):
            if not ann.validate():
                errors.append(f"Аннотация {i}: некорректные координаты - {ann}")
        return errors

    def clamp_all_to_bounds(self):
        """Привести все аннотации к допустимым границам"""
        for ann in self.annotations:
            ann.clamp_to_bounds()

    def get_class_ids(self) -> set:
        """Получить множество всех ID классов"""
        return set(ann.class_id for ann in self.annotations)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> YOLOAnnotation:
        return self.annotations[index]

    def __iter__(self):
        return iter(self.annotations)

    def __str__(self) -> str:
        return f"YOLOAnnotations({len(self.annotations)} annotations)"


def load_yolo_annotations(label_path: Union[str, Path]) -> YOLOAnnotations:
    """
    Загрузить аннотации YOLO из txt файла

    Args:
        label_path: Путь к файлу с аннотациями

    Returns:
        Объект YOLOAnnotations
    """
    label_path = Path(label_path)
    annotations = YOLOAnnotations()

    if not label_path.exists():
        return annotations

    try:
        with open(label_path, 'r', encoding=DEFAULT_ENCODING) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split()
                    if len(parts) != 5:
                        logger.warning(f"Строка {line_num} в {label_path}: ожидается 5 значений, получено {len(parts)}")
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    annotation = YOLOAnnotation(class_id, x_center, y_center, width, height)
                    annotations.add_annotation(annotation)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Ошибка парсинга строки {line_num} в {label_path}: {e}")
                    continue

    except Exception as e:
        logger.error(f"Ошибка чтения файла {label_path}: {e}")

    return annotations


def save_yolo_annotations(annotations: YOLOAnnotations, label_path: Union[str, Path]):
    """
    Сохранить аннотации YOLO в txt файл

    Args:
        annotations: Аннотации для сохранения
        label_path: Путь к файлу для сохранения
    """
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(label_path, 'w', encoding=DEFAULT_ENCODING) as f:
            for ann in annotations:
                f.write(str(ann) + '\n')
    except Exception as e:
        logger.error(f"Ошибка сохранения файла {label_path}: {e}")
        raise


def validate_yolo_dataset(images_dir: Union[str, Path], labels_dir: Union[str, Path],
                         fix_errors: bool = False) -> Dict[str, List[str]]:
    """
    Проверить корректность датасета YOLO

    Args:
        images_dir: Директория с изображениями
        labels_dir: Директория с аннотациями
        fix_errors: Автоматически исправлять ошибки

    Returns:
        Словарь с результатами валидации
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Директория с изображениями не найдена: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Директория с аннотациями не найдена: {labels_dir}")

    results = {
        'valid_images': [],
        'invalid_images': [],
        'missing_labels': [],
        'empty_labels': [],
        'annotation_errors': [],
        'total_images': 0,
        'total_annotations': 0
    }

    # Найти все изображения
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"**/*{ext}"))
        image_files.extend(images_dir.glob(f"**/*{ext.upper()}"))

    results['total_images'] = len(image_files)

    for img_path in image_files:
        # Получить соответствующий файл аннотаций
        label_path = labels_dir / f"{img_path.stem}.txt"

        # Проверить размеры изображения
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                results['invalid_images'].append(str(img_path))
                continue

            img_height, img_width = img.shape[:2]
        except Exception as e:
            logger.error(f"Ошибка чтения изображения {img_path}: {e}")
            results['invalid_images'].append(str(img_path))
            continue

        # Проверить файл аннотаций
        if not label_path.exists():
            results['missing_labels'].append(str(img_path))
            continue

        annotations = load_yolo_annotations(label_path)

        if len(annotations) == 0:
            results['empty_labels'].append(str(img_path))
            continue

        # Проверить аннотации
        errors = annotations.validate_all()
        if errors:
            results['annotation_errors'].extend([f"{img_path}: {error}" for error in errors])

            if fix_errors:
                # Исправить ошибки
                annotations.clamp_all_to_bounds()
                save_yolo_annotations(annotations, label_path)
                logger.info(f"Исправлены аннотации для {img_path}")
        else:
            results['valid_images'].append(str(img_path))

        results['total_annotations'] += len(annotations)

    return results


def get_class_distribution(labels_dir: Union[str, Path]) -> Dict[int, int]:
    """
    Получить распределение классов в датасете

    Args:
        labels_dir: Директория с аннотациями

    Returns:
        Словарь {class_id: count}
    """
    labels_dir = Path(labels_dir)
    distribution = {}

    if not labels_dir.exists():
        return distribution

    label_files = labels_dir.glob("*.txt")
    for label_file in label_files:
        annotations = load_yolo_annotations(label_file)
        for ann in annotations:
            distribution[ann.class_id] = distribution.get(ann.class_id, 0) + 1

    return distribution


def convert_yolo_to_coco(annotations: YOLOAnnotations, img_width: int, img_height: int) -> List[Dict]:
    """
    Преобразовать аннотации YOLO в формат COCO

    Args:
        annotations: Аннотации YOLO
        img_width: Ширина изображения
        img_height: Высота изображения

    Returns:
        Список аннотаций в формате COCO
    """
    coco_annotations = []

    for ann in annotations:
        x, y, width, height = ann.to_coco_bbox(img_width, img_height)
        coco_ann = {
            'id': len(coco_annotations),  # Временный ID
            'image_id': 0,  # Нужно установить извне
            'category_id': ann.class_id,
            'bbox': [x, y, width, height],
            'area': width * height,
            'iscrowd': 0
        }
        coco_annotations.append(coco_ann)

    return coco_annotations
