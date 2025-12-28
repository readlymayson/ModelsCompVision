# -*- coding: utf-8 -*-
"""
Командная строка для Video2Dataset
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
import argparse
import sys

try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

from .config import (
    DEFAULT_DATA_DIR, DEFAULT_VIDEOS_DIR, DEFAULT_IMAGES_DIR, DEFAULT_LABELS_DIR,
    DEFAULT_AUGMENTED_DIR, DEFAULT_PREVIEWS_DIR, DEFAULT_YOLO_MODEL,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IOU_THRESHOLD, DEFAULT_NUM_AUGS_PER_IMAGE,
    setup_default_directories
)
from .video_splitter import split_video, split_video_directory
from .annotation_formats import validate_yolo_dataset, get_class_distribution
from .auto_annotator import run_auto_annotation, create_visualization
from .augmentations import augment_dataset
from .yolo_integration import create_data_yaml, generate_yolo_config_from_dataset
from .dataset_viewer import run_dataset_viewer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser():
    """Создать парсер командной строки"""
    parser = argparse.ArgumentParser(
        description="Video2Dataset - инструмент для подготовки датасетов из видео под YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Разбить видео на кадры
  python -m video2dataset.cli split-video data/videos/sample.mp4 --output data/images --fps 1.0

  # Автоматическая разметка
  python -m video2dataset.cli auto-annotate --images data/images --labels data/labels

  # Аугментация датасета
  python -m video2dataset.cli augment --images data/images --labels data/labels --output-images data/augmented/images --output-labels data/augmented/labels --num-augs 3

  # Валидация аннотаций
  python -m video2dataset.cli validate-labels --images data/images --labels data/labels

  # Создание конфига для обучения
  python -m video2dataset.cli make-yolo-config --images data/images --labels data/labels --output data.yaml
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')

    # Команда split-video
    split_parser = subparsers.add_parser('split-video', help='Разбить видео на кадры JPG')
    split_parser.add_argument('input', help='Путь к видео файлу или директории')
    split_parser.add_argument('--output', '-o', required=True, help='Директория для сохранения кадров')
    split_parser.add_argument('--fps', type=float, default=1.0, help='Частота извлечения кадров (кадры в секунду)')
    split_parser.add_argument('--every-n', type=int, help='Извлекать каждый N-й кадр')
    split_parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help='Изменить размер изображений')
    split_parser.add_argument('--format', default='jpg', choices=['jpg', 'png'], help='Формат изображений')

    # Команда auto-annotate
    annotate_parser = subparsers.add_parser('auto-annotate', help='Автоматическая разметка с помощью YOLOv8')
    annotate_parser.add_argument('--images', required=True, help='Директория с изображениями')
    annotate_parser.add_argument('--labels', required=True, help='Директория для сохранения аннотаций')
    annotate_parser.add_argument('--model', default=DEFAULT_YOLO_MODEL, help='Путь к модели YOLO8')
    annotate_parser.add_argument('--conf', type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help='Порог уверенности')
    annotate_parser.add_argument('--iou', type=float, default=DEFAULT_IOU_THRESHOLD, help='Порог IoU для NMS')
    annotate_parser.add_argument('--mode', choices=['fill_missing', 'overwrite', 'backup'], default='fill_missing',
                                help='Режим работы: fill_missing (только отсутствующие), overwrite (перезаписать все), backup (с бэкапом)')

    # Команда augment
    augment_parser = subparsers.add_parser('augment', help='Аугментация датасета')
    augment_parser.add_argument('--images', required=True, help='Исходные изображения')
    augment_parser.add_argument('--labels', required=True, help='Исходные аннотации')
    augment_parser.add_argument('--output-images', required=True, help='Выходные изображения')
    augment_parser.add_argument('--output-labels', required=True, help='Выходные аннотации')
    augment_parser.add_argument('--num-augs', type=int, default=DEFAULT_NUM_AUGS_PER_IMAGE, help='Количество аугментаций на изображение')
    augment_parser.add_argument('--transforms', nargs='+', help='Список трансформаций (rotate, flip_h, brightness, etc.)')

    # Команда validate-labels
    validate_parser = subparsers.add_parser('validate-labels', help='Проверить корректность аннотаций')
    validate_parser.add_argument('--images', required=True, help='Директория с изображениями')
    validate_parser.add_argument('--labels', required=True, help='Директория с аннотациями')
    validate_parser.add_argument('--fix', action='store_true', help='Автоматически исправить ошибки')
    validate_parser.add_argument('--output', help='Файл для сохранения отчета')

    # Команда inspect-image
    inspect_parser = subparsers.add_parser('inspect-image', help='Просмотр информации об изображении и аннотациях')
    inspect_parser.add_argument('image_path', help='Путь к изображению')
    inspect_parser.add_argument('--labels', help='Директория с аннотациями')
    inspect_parser.add_argument('--visualize', action='store_true', help='Создать визуализацию с bounding boxes')
    inspect_parser.add_argument('--output', help='Путь для сохранения визуализации')

    # Команда make-yolo-config
    config_parser = subparsers.add_parser('make-yolo-config', help='Создать data.yaml для обучения YOLOv8')
    config_parser.add_argument('--images', required=True, help='Директория с изображениями')
    config_parser.add_argument('--labels', required=True, help='Директория с аннотациями')
    config_parser.add_argument('--output', required=True, help='Путь для сохранения data.yaml')
    config_parser.add_argument('--classes', nargs='+', help='Имена классов')
    config_parser.add_argument('--split-dataset', action='store_true', help='Разделить датасет на train/val/test папки')


    # Команда setup
    setup_parser = subparsers.add_parser('setup', help='Создать структуру директорий по умолчанию')
    setup_parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR), help='Базовая директория для данных')

    # Команда view-dataset
    view_parser = subparsers.add_parser('view-dataset', help='Запустить веб-интерфейс для просмотра датасета')
    view_parser.add_argument('--images', required=True, help='Директория с изображениями')
    view_parser.add_argument('--labels', required=True, help='Директория с аннотациями')
    view_parser.add_argument('--host', default='127.0.0.1', help='Хост для запуска сервера (по умолчанию: 127.0.0.1)')
    view_parser.add_argument('--port', type=int, default=5000, help='Порт для запуска сервера (по умолчанию: 5000)')
    view_parser.add_argument('--debug', action='store_true', help='Включить режим отладки')
    view_parser.add_argument('--classes', nargs='+', help='Имена классов для отображения')

    return parser


def cmd_split_video(args):
    """Обработчик команды split-video"""
    input_path = Path(args.input)
    output_dir = Path(args.output)

    resize = None
    if args.resize:
        resize = tuple(args.resize)

    if input_path.is_file():
        # Разбить одно видео
        logger.info(f"Разбиение видео: {input_path}")
        saved_paths = split_video(
            input_path, output_dir, args.fps, args.every_n, resize, args.format
        )
        logger.info(f"Извлечено {len(saved_paths)} кадров")

    elif input_path.is_dir():
        # Обработать директорию с видео
        logger.info(f"Обработка директории: {input_path}")
        results = split_video_directory(
            input_path, output_dir, args.fps, args.every_n, resize, args.format
        )
        logger.info(f"Обработано {results['total_videos']} видео, извлечено {results['total_frames']} кадров")

    else:
        logger.error(f"Путь не найден: {input_path}")
        return 1

    return 0


def cmd_auto_annotate(args):
    """Обработчик команды auto-annotate"""
    try:
        stats = run_auto_annotation(
            args.images, args.labels, args.model, args.conf, args.iou, args.mode
        )

        logger.info("Автоматическая разметка завершена:")
        logger.info(f"  Всего изображений: {stats['total_images']}")
        logger.info(f"  Обработано: {stats['processed']}")
        logger.info(f"  Пропущено: {stats['skipped']}")
        logger.info(f"  Ошибок: {stats['errors']}")

        return 0

    except Exception as e:
        logger.error(f"Ошибка автоматической разметки: {e}")
        return 1


def cmd_augment(args):
    """Обработчик команды augment"""
    try:
        stats = augment_dataset(
            args.images, args.labels, args.output_images, args.output_labels,
            args.num_augs, args.transforms
        )

        logger.info("Аугментация завершена:")
        logger.info(f"  Всего изображений: {stats['total_images']}")
        logger.info(f"  Аугментировано: {stats['augmented']}")
        logger.info(f"  Скопировано оригиналов: {stats['copied']}")
        logger.info(f"  Ошибок: {stats['errors']}")

        return 0

    except Exception as e:
        logger.error(f"Ошибка аугментации: {e}")
        return 1


def cmd_validate_labels(args):
    """Обработчик команды validate-labels"""
    try:
        results = validate_yolo_dataset(args.images, args.labels, args.fix)

        logger.info("Результаты валидации:")
        logger.info(f"  Всего изображений: {results['total_images']}")
        logger.info(f"  Корректных изображений: {len(results['valid_images'])}")
        logger.info(f"  Изображений без аннотаций: {len(results['missing_labels'])}")
        logger.info(f"  Пустых аннотаций: {len(results['empty_labels'])}")
        logger.info(f"  Ошибок в аннотациях: {len(results['annotation_errors'])}")

        if results['annotation_errors']:
            logger.info("Ошибки аннотаций:")
            for error in results['annotation_errors'][:10]:  # Показать первые 10
                logger.info(f"  {error}")
            if len(results['annotation_errors']) > 10:
                logger.info(f"  ... и еще {len(results['annotation_errors']) - 10} ошибок")

        if args.output:
            # Сохранить отчет
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Отчет сохранен: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Ошибка валидации: {e}")
        return 1


def cmd_inspect_image(args):
    """Обработчик команды inspect-image"""
    from .annotation_formats import load_yolo_annotations
    import cv2

    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"Изображение не найдено: {image_path}")
        return 1

    # Загрузить изображение
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Не удалось загрузить изображение: {image_path}")
        return 1

    height, width = img.shape[:2]
    logger.info(f"Изображение: {width}x{height}")

    # Загрузить аннотации
    annotations = None
    if args.labels:
        labels_dir = Path(args.labels)
        label_path = labels_dir / f"{image_path.stem}.txt"
        annotations = load_yolo_annotations(label_path)

        logger.info(f"Аннотаций: {len(annotations) if annotations else 0}")

        if annotations and len(annotations) > 0:
            class_counts = {}
            for ann in annotations:
                class_counts[ann.class_id] = class_counts.get(ann.class_id, 0) + 1

            logger.info("Распределение по классам:")
            for class_id, count in sorted(class_counts.items()):
                logger.info(f"  Класс {class_id}: {count} объектов")

    # Создать визуализацию
    if args.visualize:
        if annotations:
            try:
                output_path = args.output or image_path.parent / f"{image_path.stem}_annotated{image_path.suffix}"
                vis_path = create_visualization(image_path, annotations, output_path)
                logger.info(f"Визуализация сохранена: {vis_path}")
            except Exception as e:
                logger.error(f"Ошибка создания визуализации: {e}")
        else:
            logger.warning("Нет аннотаций для визуализации")

    return 0


def cmd_make_yolo_config(args):
    """Обработчик команды make-yolo-config"""
    try:
        if args.split_dataset:
            # Создать полный конфиг с разделением
            yaml_path = generate_yolo_config_from_dataset(
                args.images, args.labels, args.output, args.classes
            )
        else:
            # Создать простой конфиг
            yaml_path = create_data_yaml(
                args.images, args.labels, args.output, args.classes
            )

        logger.info(f"Конфигурационный файл создан: {yaml_path}")

        # Показать распределение классов
        class_dist = get_class_distribution(args.labels)
        if class_dist:
            logger.info("Распределение классов в датасете:")
            for class_id, count in sorted(class_dist.items()):
                logger.info(f"  Класс {class_id}: {count} объектов")

        return 0

    except Exception as e:
        logger.error(f"Ошибка создания конфига: {e}")
        return 1


def cmd_setup(args):
    """Обработчик команды setup"""
    data_dir = Path(args.data_dir)

    try:
        setup_default_directories()
        logger.info("Структура директорий создана:")
        logger.info(f"  Видео: {DEFAULT_VIDEOS_DIR}")
        logger.info(f"  Изображения: {DEFAULT_IMAGES_DIR}")
        logger.info(f"  Аннотации: {DEFAULT_LABELS_DIR}")
        logger.info(f"  Аугментированные: {DEFAULT_AUGMENTED_DIR}")
        logger.info(f"  Превью: {DEFAULT_PREVIEWS_DIR}")

        return 0

    except Exception as e:
        logger.error(f"Ошибка создания структуры: {e}")
        return 1


def cmd_view_dataset(args):
    """Обработчик команды view-dataset"""
    try:
        # Подготовить словарь с именами классов
        class_names = None
        if args.classes:
            class_names = {i: name for i, name in enumerate(args.classes)}

        # Запустить веб-интерфейс
        run_dataset_viewer(
            images_dir=args.images,
            labels_dir=args.labels,
            host=args.host,
            port=args.port,
            debug=args.debug,
            class_names=class_names
        )

        return 0

    except Exception as e:
        logger.error(f"Ошибка запуска веб-интерфейса: {e}")
        return 1


def main():
    """Главная функция CLI"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Настроить директории по умолчанию
    setup_default_directories()

    # Обработать команду
    command_handlers = {
        'split-video': cmd_split_video,
        'auto-annotate': cmd_auto_annotate,
        'augment': cmd_augment,
        'validate-labels': cmd_validate_labels,
        'inspect-image': cmd_inspect_image,
        'make-yolo-config': cmd_make_yolo_config,
        'setup': cmd_setup,
        'view-dataset': cmd_view_dataset,
    }

    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        logger.error(f"Неизвестная команда: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
