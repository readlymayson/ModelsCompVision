# -*- coding: utf-8 -*-
"""
Модуль для автоматической разметки изображений с помощью YOLOv8
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Dict
import shutil

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

import cv2
import numpy as np
import tqdm

from .annotation_formats import YOLOAnnotations, YOLOAnnotation, save_yolo_annotations
from .config import DEFAULT_YOLO_MODEL, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IOU_THRESHOLD

logger = logging.getLogger(__name__)


class AutoAnnotator:
    """Класс для автоматической разметки изображений с помощью YOLOv8"""

    def __init__(self, model_path: str = DEFAULT_YOLO_MODEL,
                 conf_thres: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 iou_thres: float = DEFAULT_IOU_THRESHOLD):
        """
        Инициализация автоаннотатора

        Args:
            model_path: Путь к модели YOLOv8
            conf_thres: Порог уверенности для детекции
            iou_thres: Порог IoU для NMS
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics не установлен. Установите его командой: pip install ultralytics")

        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Загрузить модель
        try:
            self.model = YOLO(model_path)
            logger.info(f"Модель YOLO загружена: {model_path}")
        except Exception as e:
            raise ValueError(f"Не удалось загрузить модель {model_path}: {e}")

    def predict_image(self, image_path: Union[str, Path]) -> YOLOAnnotations:
        """
        Предсказать аннотации для одного изображения

        Args:
            image_path: Путь к изображению

        Returns:
            Аннотации YOLOAnnotations
        """
        image_path = Path(image_path)

        # Выполнить предсказание
        try:
            results = self.model(image_path, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        except Exception as e:
            logger.error(f"Ошибка предсказания для {image_path}: {e}")
            return YOLOAnnotations()

        if not results or len(results) == 0:
            return YOLOAnnotations()

        result = results[0]  # Результаты для первого (единственного) изображения

        annotations = YOLOAnnotations()

        # Извлечь bounding boxes
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes

            for box in boxes:
                try:
                    # Координаты в формате [x1, y1, x2, y2]
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    conf = box.conf.cpu().numpy().item()
                    class_id = int(box.cls.cpu().numpy().item())

                    # Получить размеры изображения
                    img = cv2.imread(str(image_path))
                    if img is None:
                        logger.warning(f"Не удалось прочитать изображение: {image_path}")
                        continue

                    img_height, img_width = img.shape[:2]

                    # Создать аннотацию YOLO из координат в пикселях
                    yolo_ann = YOLOAnnotation.from_pixels(
                        class_id, int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]),
                        img_width, img_height
                    )

                    annotations.add_annotation(yolo_ann)

                except Exception as e:
                    logger.warning(f"Ошибка обработки bbox: {e}")
                    continue

        return annotations

    def predict_batch(self, image_paths: List[Union[str, Path]]) -> Dict[str, YOLOAnnotations]:
        """
        Предсказать аннотации для батча изображений

        Args:
            image_paths: Список путей к изображениям

        Returns:
            Словарь {image_path: YOLOAnnotations}
        """
        results = {}

        # Обработать каждое изображение отдельно для лучшего контроля ошибок
        for img_path in tqdm.tqdm(image_paths, desc="Автоматическая разметка"):
            annotations = self.predict_image(img_path)
            results[str(img_path)] = annotations

        return results

    def run_auto_annotation(self, images_dir: Union[str, Path], labels_dir: Union[str, Path],
                           mode: str = 'fill_missing', backup_existing: bool = False) -> Dict[str, int]:
        """
        Запустить автоматическую разметку для директории изображений

        Args:
            images_dir: Директория с изображениями
            labels_dir: Директория для сохранения аннотаций
            mode: Режим работы ('fill_missing', 'overwrite', 'backup')
            backup_existing: Создать бэкап существующих аннотаций

        Returns:
            Статистика обработки
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        if not images_dir.exists():
            raise FileNotFoundError(f"Директория с изображениями не найдена: {images_dir}")

        labels_dir.mkdir(parents=True, exist_ok=True)

        # Найти все изображения
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            path for path in images_dir.rglob('*')
            if path.suffix.lower() in image_extensions
        ]

        if not image_files:
            logger.warning(f"Изображения не найдены в директории: {images_dir}")
            return {'total_images': 0, 'processed': 0, 'skipped': 0, 'errors': 0}

        logger.info(f"Найдено {len(image_files)} изображений для обработки")

        stats = {
            'total_images': len(image_files),
            'processed': 0,
            'skipped': 0,
            'errors': 0
        }

        # Создать директорию для бэкапа если нужно
        backup_dir = None
        if backup_existing and mode in ['overwrite', 'backup']:
            backup_dir = labels_dir.parent / f"{labels_dir.name}_backup"
            backup_dir.mkdir(exist_ok=True)
            logger.info(f"Бэкап существующих аннотаций будет сохранен в: {backup_dir}")

        for img_path in tqdm.tqdm(image_files, desc="Обработка изображений"):
            try:
                # Путь к файлу аннотаций
                label_path = labels_dir / f"{img_path.stem}.txt"

                # Проверить существует ли уже аннотация
                has_existing = label_path.exists()

                should_process = False
                if mode == 'fill_missing':
                    should_process = not has_existing
                elif mode == 'overwrite':
                    should_process = True
                elif mode == 'backup':
                    should_process = True

                if not should_process:
                    stats['skipped'] += 1
                    continue

                # Создать бэкап если нужно
                if backup_existing and has_existing and mode in ['overwrite', 'backup']:
                    backup_path = backup_dir / label_path.name
                    shutil.copy2(label_path, backup_path)

                # Выполнить предсказание
                annotations = self.predict_image(img_path)

                # Сохранить аннотации
                save_yolo_annotations(annotations, label_path)

                stats['processed'] += 1

            except Exception as e:
                logger.error(f"Ошибка обработки {img_path}: {e}")
                stats['errors'] += 1
                continue

        logger.info(f"Обработка завершена. Статистика: {stats}")
        return stats

    def get_model_info(self) -> Dict:
        """
        Получить информацию о загруженной модели

        Returns:
            Информация о модели
        """
        try:
            return {
                'model_path': self.model_path,
                'conf_thres': self.conf_thres,
                'iou_thres': self.iou_thres,
                'model_names': self.model.names if hasattr(self.model, 'names') else None
            }
        except Exception as e:
            return {'error': str(e)}


def create_visualization(image_path: Union[str, Path], annotations: YOLOAnnotations,
                        output_path: Optional[Union[str, Path]] = None,
                        class_names: Optional[Dict[int, str]] = None) -> str:
    """
    Создать визуализацию предсказаний на изображении

    Args:
        image_path: Путь к изображению
        annotations: Аннотации для визуализации
        output_path: Путь для сохранения результата (если None, будет создан автоматически)
        class_names: Словарь {class_id: class_name} для подписей

    Returns:
        Путь к сохраненному изображению
    """
    image_path = Path(image_path)

    # Загрузить изображение
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    img_height, img_width = img.shape[:2]

    # Нарисовать bounding boxes
    for ann in annotations:
        # Получить координаты в пикселях
        x1, y1, x2, y2 = ann.to_pixels(img_width, img_height)

        # Цвет для класса (разные цвета для разных классов)
        color = (0, 255, 0)  # Зеленый по умолчанию
        if ann.class_id is not None:
            # Генерировать цвет на основе class_id
            np.random.seed(ann.class_id)
            color = tuple(np.random.randint(0, 255, 3).tolist())

        # Нарисовать прямоугольник
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Подпись с классом и уверенностью
        label = f"Class {ann.class_id}"
        if class_names and ann.class_id in class_names:
            label = class_names[ann.class_id]

        # Фон для текста
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)

        # Текст
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Сохранить результат
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_annotated{image_path.suffix}"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), img)
    if not success:
        raise IOError(f"Не удалось сохранить изображение: {output_path}")

    return str(output_path)


def run_auto_annotation(images_dir: Union[str, Path], labels_dir: Union[str, Path],
                       model_path: str = DEFAULT_YOLO_MODEL,
                       conf_thres: float = DEFAULT_CONFIDENCE_THRESHOLD,
                       iou_thres: float = DEFAULT_IOU_THRESHOLD,
                       mode: str = 'fill_missing') -> Dict[str, int]:
    """
    Удобная функция для запуска автоматической разметки

    Args:
        images_dir: Директория с изображениями
        labels_dir: Директория для аннотаций
        model_path: Путь к модели YOLO
        conf_thres: Порог уверенности
        iou_thres: Порог IoU
        mode: Режим работы ('fill_missing', 'overwrite', 'backup')

    Returns:
        Статистика обработки
    """
    annotator = AutoAnnotator(model_path, conf_thres, iou_thres)
    return annotator.run_auto_annotation(images_dir, labels_dir, mode, backup_existing=(mode == 'backup'))
