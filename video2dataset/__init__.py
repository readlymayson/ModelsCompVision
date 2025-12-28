# -*- coding: utf-8 -*-
"""
Video2Dataset - инструмент для подготовки датасетов из видео под YOLOv8

Предоставляет функционал для:
- Разбиения видео на кадры JPG
- Автоматической разметки с помощью YOLOv8
- Аугментации изображений и аннотаций
- Валидации аннотаций и подготовки к обучению
"""

__version__ = "0.1.0"
__author__ = "Video2Dataset Team"

# Основные импорты для удобства использования
from .video_splitter import VideoSplitter, split_video, split_video_directory
from .annotation_formats import YOLOAnnotations, YOLOAnnotation, validate_yolo_dataset
from .auto_annotator import AutoAnnotator, run_auto_annotation
from .augmentations import YOLOAugmenter, augment_dataset
from .yolo_integration import create_data_yaml, generate_yolo_config_from_dataset
from .dataset_viewer import create_dataset_viewer_app, run_dataset_viewer

__all__ = [
    # Основные классы
    'VideoSplitter',
    'YOLOAnnotations',
    'YOLOAnnotation',
    'AutoAnnotator',
    'YOLOAugmenter',

    # Удобные функции
    'split_video',
    'split_video_directory',
    'validate_yolo_dataset',
    'run_auto_annotation',
    'augment_dataset',
    'create_data_yaml',
    'generate_yolo_config_from_dataset',
    'create_dataset_viewer_app',
    'run_dataset_viewer',
]
