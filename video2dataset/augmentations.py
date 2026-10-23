# -*- coding: utf-8 -*-
"""
Модуль для аугментации изображений и аннотаций
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import random
import shutil

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None

import cv2
import numpy as np
import tqdm

from .annotation_formats import YOLOAnnotations, YOLOAnnotation, load_yolo_annotations, save_yolo_annotations
from .config import DEFAULT_NUM_AUGS_PER_IMAGE, DEFAULT_AUGMENTATION_PROBABILITY

logger = logging.getLogger(__name__)


class YOLOAugmenter:
    """Класс для аугментации изображений с поддержкой YOLO аннотаций"""

    def __init__(self, transforms: Optional[List] = None, probability: float = DEFAULT_AUGMENTATION_PROBABILITY):
        """
        Инициализация аугментатора

        Args:
            transforms: Список трансформаций Albumentations
            probability: Вероятность применения аугментации
        """
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations не установлен. Установите его командой: pip install albumentations")

        if transforms is None:
            # Стандартный набор аугментаций
            transforms = self._get_default_transforms()

        self.transforms = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',  # Формат YOLO: x_center, y_center, width, height (нормализованные)
                label_fields=['class_ids'],
                min_visibility=0.1  # Минимальная видимость bbox после трансформации
            )
        )

        self.probability = probability

    def _get_default_transforms(self) -> List:
        """Получить стандартный набор аугментаций"""
        return [
            # Геометрические трансформации
            A.Rotate(limit=15, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.1),

            # Изменение яркости и контраста
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),

            # Gaussian шум
            A.GaussNoise(var_limit=(10, 50), p=0.2),

            # Размытие
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),

            # Изменение насыщенности и оттенка
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2),

            # Масштабирование и кроп
            A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.3),

            # Affine трансформации
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-10, 10), p=0.2),
        ]

    def augment_image(self, image: np.ndarray, annotations: YOLOAnnotations) -> Tuple[np.ndarray, YOLOAnnotations]:
        """
        Применить аугментацию к изображению и аннотациям

        Args:
            image: Изображение в формате numpy array
            annotations: Аннотации YOLOAnnotations

        Returns:
            Кортеж (augmented_image, augmented_annotations)
        """
        if random.random() > self.probability:
            # Возвратить оригинал без изменений
            return image.copy(), YOLOAnnotations(annotations.annotations.copy())

        # Подготовить данные для Albumentations
        bboxes = []
        class_ids = []

        for ann in annotations:
            bboxes.append([ann.x_center, ann.y_center, ann.width, ann.height])
            class_ids.append(ann.class_id)

        # Применить аугментацию
        try:
            transformed = self.transforms(
                image=image,
                bboxes=bboxes,
                class_ids=class_ids
            )

            augmented_image = transformed['image']
            augmented_bboxes = transformed['bboxes']
            augmented_class_ids = transformed['class_ids']

            # Создать новые аннотации
            augmented_annotations = YOLOAnnotations()
            for i, bbox in enumerate(augmented_bboxes):
                if len(bbox) >= 4:  # Убедиться что bbox валиден
                    ann = YOLOAnnotation(
                        class_id=augmented_class_ids[i],
                        x_center=bbox[0],
                        y_center=bbox[1],
                        width=bbox[2],
                        height=bbox[3]
                    )
                    augmented_annotations.add_annotation(ann)

        except Exception as e:
            logger.warning(f"Ошибка аугментации: {e}. Возвращаю оригинал.")
            return image.copy(), YOLOAnnotations(annotations.annotations.copy())

        return augmented_image, augmented_annotations

    def augment_image_file(self, image_path: Union[str, Path], annotations: YOLOAnnotations,
                          output_image_path: Union[str, Path],
                          output_annotations_path: Union[str, Path]) -> bool:
        """
        Аугментировать изображение из файла и сохранить результат

        Args:
            image_path: Путь к исходному изображению
            annotations: Аннотации для изображения
            output_image_path: Путь для сохранения аугментированного изображения
            output_annotations_path: Путь для сохранения аугментированных аннотаций

        Returns:
            True если аугментация прошла успешно
        """
        image_path = Path(image_path)
        output_image_path = Path(output_image_path)
        output_annotations_path = Path(output_annotations_path)

        # Загрузить изображение
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Не удалось загрузить изображение: {image_path}")
            return False

        # Конвертировать BGR в RGB для Albumentations
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Применить аугментацию
        augmented_image, augmented_annotations = self.augment_image(image, annotations)

        # Конвертировать обратно в BGR для сохранения
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

        # Создать директории
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        output_annotations_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохранить изображение
        success = cv2.imwrite(str(output_image_path), augmented_image)
        if not success:
            logger.error(f"Не удалось сохранить изображение: {output_image_path}")
            return False

        # Сохранить аннотации
        save_yolo_annotations(augmented_annotations, output_annotations_path)

        return True

    def augment_dataset(self, images_dir: Union[str, Path], labels_dir: Union[str, Path],
                       output_images_dir: Union[str, Path], output_labels_dir: Union[str, Path],
                       num_augs_per_image: int = DEFAULT_NUM_AUGS_PER_IMAGE,
                       copy_original: bool = True) -> Dict[str, int]:
        """
        Аугментировать весь датасет

        Args:
            images_dir: Директория с исходными изображениями
            labels_dir: Директория с исходными аннотациями
            output_images_dir: Директория для аугментированных изображений
            output_labels_dir: Директория для аугментированных аннотаций
            num_augs_per_image: Количество аугментаций на изображение
            copy_original: Копировать оригинальные изображения в выходную директорию

        Returns:
            Статистика аугментации
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        output_images_dir = Path(output_images_dir)
        output_labels_dir = Path(output_labels_dir)

        # Проверить существование директорий
        if not images_dir.exists():
            raise FileNotFoundError(f"Директория с изображениями не найдена: {images_dir}")

        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        # Найти все изображения
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"**/*{ext}"))
            image_files.extend(images_dir.glob(f"**/*{ext.upper()}"))

        if not image_files:
            logger.warning(f"Изображения не найдены в директории: {images_dir}")
            return {'total_images': 0, 'augmented': 0, 'copied': 0, 'errors': 0}

        logger.info(f"Найдено {len(image_files)} изображений для аугментации")

        stats = {
            'total_images': len(image_files),
            'augmented': 0,
            'copied': 0,
            'errors': 0
        }

        for img_path in tqdm.tqdm(image_files, desc="Аугментация датасета"):
            try:
                # Путь к аннотациям
                label_path = labels_dir / f"{img_path.stem}.txt"

                # Загрузить аннотации
                annotations = load_yolo_annotations(label_path)

                # Копировать оригинал если нужно
                if copy_original:
                    # Копировать изображение
                    orig_img_output = output_images_dir / f"{img_path.stem}_original{img_path.suffix}"
                    shutil.copy2(img_path, orig_img_output)

                    # Копировать аннотации
                    orig_label_output = output_labels_dir / f"{img_path.stem}_original.txt"
                    if label_path.exists():
                        shutil.copy2(label_path, orig_label_output)
                    else:
                        # Создать пустой файл аннотаций
                        orig_label_output.touch()

                    stats['copied'] += 1

                # Создать аугментации
                for aug_idx in range(num_augs_per_image):
                    # Генерировать имена файлов
                    aug_img_name = f"{img_path.stem}_aug_{aug_idx:06d}{img_path.suffix}"
                    aug_img_path = output_images_dir / aug_img_name

                    aug_label_name = f"{img_path.stem}_aug_{aug_idx:06d}.txt"
                    aug_label_path = output_labels_dir / aug_label_name

                    # Применить аугментацию
                    success = self.augment_image_file(
                        img_path, annotations, aug_img_path, aug_label_path
                    )

                    if success:
                        stats['augmented'] += 1
                    else:
                        stats['errors'] += 1

            except Exception as e:
                logger.error(f"Ошибка обработки {img_path}: {e}")
                stats['errors'] += 1
                continue

        logger.info(f"Аугментация завершена. Статистика: {stats}")
        return stats


def create_custom_augmenter(transform_list: List[str]) -> YOLOAugmenter:
    """
    Создать аугментатор с пользовательским набором трансформаций

    Args:
        transform_list: Список названий трансформаций

    Returns:
        YOLOAugmenter с указанными трансформациями
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations не установлен")

    transforms = []

    # Словарь доступных трансформаций
    available_transforms = {
        'rotate': lambda: A.Rotate(limit=15, p=0.5),
        'flip_h': lambda: A.HorizontalFlip(p=0.5),
        'flip_v': lambda: A.VerticalFlip(p=0.5),
        'brightness': lambda: A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        'noise': lambda: A.GaussNoise(var_limit=(10, 50), p=0.5),
        'blur': lambda: A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        'hue': lambda: A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        'crop': lambda: A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        'affine': lambda: A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-10, 10), p=0.5),
    }

    for transform_name in transform_list:
        if transform_name in available_transforms:
            transforms.append(available_transforms[transform_name]())
        else:
            logger.warning(f"Неизвестная трансформация: {transform_name}")

    if not transforms:
        logger.warning("Не указаны валидные трансформации, использую стандартный набор")
        return YOLOAugmenter()

    return YOLOAugmenter(transforms)


def augment_dataset(images_dir: Union[str, Path], labels_dir: Union[str, Path],
                   output_images_dir: Union[str, Path], output_labels_dir: Union[str, Path],
                   num_augs_per_image: int = DEFAULT_NUM_AUGS_PER_IMAGE,
                   transforms: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Удобная функция для аугментации датасета

    Args:
        images_dir: Исходные изображения
        labels_dir: Исходные аннотации
        output_images_dir: Выходные изображения
        output_labels_dir: Выходные аннотации
        num_augs_per_image: Количество аугментаций на изображение
        transforms: Список трансформаций (если None, используется стандартный набор)

    Returns:
        Статистика аугментации
    """
    if transforms:
        augmenter = create_custom_augmenter(transforms)
    else:
        augmenter = YOLOAugmenter()

    return augmenter.augment_dataset(
        images_dir, labels_dir, output_images_dir, output_labels_dir,
        num_augs_per_image, copy_original=True
    )
