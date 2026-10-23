# -*- coding: utf-8 -*-
"""
Dataset Viewer - веб-интерфейс для просмотра датасетов YOLO
"""

import os
import json
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import unquote
from flask import Flask, render_template, request, jsonify, send_file, abort
from PIL import Image

# Импорты для визуализации
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .annotation_formats import (
    load_yolo_annotations,
    get_class_distribution,
    validate_yolo_dataset
)
from .config import DEFAULT_ENCODING


def create_dataset_viewer_app(images_dir: Union[str, Path],
                             labels_dir: Union[str, Path],
                             class_names: Optional[Dict[int, str]] = None) -> Flask:
    """
    Создать Flask приложение для просмотра датасета

    Args:
        images_dir: Директория с изображениями
        labels_dir: Директория с аннотациями
        class_names: Словарь {class_id: class_name}

    Returns:
        Flask приложение
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Директория с изображениями не найдена: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Директория с аннотациями не найдена: {labels_dir}")

    app = Flask(__name__,
                template_folder=str(Path(__file__).parent / 'templates'),
                static_folder=str(Path(__file__).parent.parent / 'static'))

    # Глобальные переменные приложения
    app.config['IMAGES_DIR'] = images_dir
    app.config['LABELS_DIR'] = labels_dir
    app.config['CLASS_NAMES'] = class_names or {}

    @app.route('/')
    def index():
        """Главная страница с галереей изображений"""
        return render_template('dataset_viewer/index.html')

    @app.route('/image/<path:image_name>')
    def image_view(image_name):
        """Страница просмотра отдельного изображения"""
        return render_template('dataset_viewer/image.html', image_name=image_name)

    @app.route('/stats')
    def stats_view():
        """Страница статистики датасета"""
        return render_template('dataset_viewer/stats.html')

    @app.route('/api/images/count')
    def api_images_count():
        """API: Получить количество изображений для фильтров (создает кэш для синхронизации)"""
        try:
            search = request.args.get('search', '').strip()
            class_filter = request.args.get('class_filter', '').strip()

            # Ключ кэша для текущих параметров фильтрации
            cache_key = f"{search}_{class_filter}"

            # Использовать кэш приложения для хранения списков изображений
            if not hasattr(app, 'image_cache'):
                app.image_cache = {}

            # Получить или создать отфильтрованный список изображений
            if cache_key not in app.image_cache:
                # Найти все изображения (исключая дубликаты)
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                image_files = set()  # Используем set для исключения дублей

                for ext in image_extensions:
                    # Ищем файлы с расширением в нижнем регистре
                    image_files.update(images_dir.glob(f"**/*{ext}"))
                    # Ищем файлы с расширением в верхнем регистре
                    image_files.update(images_dir.glob(f"**/*{ext.upper()}"))

                # Преобразуем set обратно в список для дальнейшей обработки
                image_files = list(image_files)

                # Фильтрация по поиску
                if search:
                    image_files = [f for f in image_files if search.lower() in f.name.lower()]

                # Фильтрация по классу
                if class_filter:
                    try:
                        filter_class_id = int(class_filter)
                        filtered_files = []
                        for img_path in image_files:
                            label_path = labels_dir / f"{img_path.stem}.txt"
                            annotations = load_yolo_annotations(label_path)
                            if any(ann.class_id == filter_class_id for ann in annotations):
                                filtered_files.append(img_path)
                        image_files = filtered_files
                    except ValueError:
                        pass  # Игнорировать некорректный class_filter

                # Сортировка по имени файла
                image_files.sort(key=lambda x: x.name)

                # Сохранить в кэш
                app.image_cache[cache_key] = image_files

            image_files = app.image_cache[cache_key]

            return jsonify({'count': len(image_files)})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/images')
    def api_images():
        """API: Получить список изображений с пагинацией"""
        try:
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 50))
            search = request.args.get('search', '').strip()
            class_filter = request.args.get('class_filter', '').strip()

            # Ключ кэша для текущих параметров фильтрации
            cache_key = f"{search}_{class_filter}"

            # Использовать кэш приложения для хранения списка файлов
            if not hasattr(app, 'image_cache'):
                app.image_cache = {}

            # Получить или создать отфильтрованный список изображений
            if cache_key not in app.image_cache:
                # Найти все изображения (исключая дубликаты)
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                image_files = set()  # Используем set для исключения дублей

                for ext in image_extensions:
                    # Ищем файлы с расширением в нижнем регистре
                    image_files.update(images_dir.glob(f"**/*{ext}"))
                    # Ищем файлы с расширением в верхнем регистре
                    image_files.update(images_dir.glob(f"**/*{ext.upper()}"))

                # Преобразуем set обратно в список для дальнейшей обработки
                image_files = list(image_files)

                # Фильтрация по поиску
                if search:
                    image_files = [f for f in image_files if search.lower() in f.name.lower()]

                # Фильтрация по классу
                if class_filter:
                    try:
                        filter_class_id = int(class_filter)
                        filtered_files = []
                        for img_path in image_files:
                            label_path = labels_dir / f"{img_path.stem}.txt"
                            annotations = load_yolo_annotations(label_path)
                            if any(ann.class_id == filter_class_id for ann in annotations):
                                filtered_files.append(img_path)
                        image_files = filtered_files
                    except ValueError:
                        pass  # Игнорировать некорректный class_filter

                # Сортировка по имени файла
                image_files.sort(key=lambda x: x.name)

                # Сохранить в кэш
                app.image_cache[cache_key] = image_files
            else:
                image_files = app.image_cache[cache_key]

            # Пагинация
            total_images = len(image_files)
            start_idx = (page - 1) * per_page
            end_idx = min(start_idx + per_page, total_images)
            page_files = image_files[start_idx:end_idx]

            # Подготовить данные для ответа
            images_data = []
            for img_path in page_files:
                label_path = labels_dir / f"{img_path.stem}.txt"
                annotations = load_yolo_annotations(label_path)

                # Получить превью (base64) только для текущей страницы
                try:
                    with Image.open(img_path) as img:
                        img.thumbnail((200, 200))
                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG', quality=70)
                        thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                except Exception:
                    thumbnail_b64 = None

                images_data.append({
                    'name': img_path.name,
                    'path': str(img_path.relative_to(images_dir)),
                    'annotations_count': len(annotations),
                    'classes': list(set(ann.class_id for ann in annotations)),
                    'thumbnail': f"data:image/jpeg;base64,{thumbnail_b64}" if thumbnail_b64 else None
                })

            return jsonify({
                'images': images_data,
                'total': total_images,
                'page': page,
                'per_page': per_page,
                'total_pages': (total_images + per_page - 1) // per_page
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/image/<path:image_name>')
    def api_image(image_name):
        """API: Получить изображение"""
        try:
            # Декодировать URL
            image_name = unquote(image_name)

            image_path = (images_dir / image_name).resolve()
            # Защита от path traversal: проверяем, что файл внутри images_dir
            if not image_path.is_relative_to(images_dir.resolve()):
                return jsonify({'error': 'Доступ запрещён'}), 403

            if not image_path.exists():
                return jsonify({'error': f'Изображение не найдено: {image_path}'}), 404

            return send_file(str(image_path), mimetype='image/jpeg')
        except Exception as e:
            return jsonify({'error': f'Ошибка загрузки изображения: {str(e)}'}), 500

    @app.route('/api/annotations/<path:image_name>')
    def api_annotations(image_name):
        """API: Получить аннотации для изображения"""
        try:
            # Декодировать URL
            image_name = unquote(image_name)

            label_path = labels_dir / f"{Path(image_name).stem}.txt"
            annotations = load_yolo_annotations(label_path)

            annotations_data = []
            for ann in annotations:
                annotations_data.append({
                    'class_id': ann.class_id,
                    'class_name': app.config['CLASS_NAMES'].get(ann.class_id, f'class_{ann.class_id}'),
                    'x_center': ann.x_center,
                    'y_center': ann.y_center,
                    'width': ann.width,
                    'height': ann.height
                })

            return jsonify({
                'annotations': annotations_data,
                'count': len(annotations_data)
            })

        except Exception as e:
            return jsonify({'error': f'Ошибка загрузки аннотаций: {str(e)}'}), 500

    @app.route('/api/stats')
    def api_stats():
        """API: Получить статистику датасета"""
        try:
            # Валидация датасета
            validation_results = validate_yolo_dataset(images_dir, labels_dir)

            # Распределение классов
            class_distribution = get_class_distribution(labels_dir)

            # Статистика классов с именами
            class_stats = []
            for class_id, count in sorted(class_distribution.items()):
                class_stats.append({
                    'id': class_id,
                    'name': app.config['CLASS_NAMES'].get(class_id, f'class_{class_id}'),
                    'count': count
                })

            stats = {
                'total_images': validation_results['total_images'],
                'valid_images': len(validation_results['valid_images']),
                'total_annotations': validation_results['total_annotations'],
                'missing_labels': len(validation_results['missing_labels']),
                'empty_labels': len(validation_results['empty_labels']),
                'annotation_errors': len(validation_results['annotation_errors']),
                'class_distribution': class_stats,
                'class_names': app.config['CLASS_NAMES']
            }

            return jsonify(stats)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/visualize/<path:image_name>')
    def api_visualize(image_name):
        """API: Получить визуализацию изображения с bounding boxes"""
        try:
            if not CV2_AVAILABLE:
                return jsonify({'error': 'OpenCV не доступен для создания визуализации'}), 500

            # Декодировать URL
            image_name = unquote(image_name)

            image_path = (images_dir / image_name).resolve()
            # Защита от path traversal: проверяем, что файл внутри images_dir
            if not image_path.is_relative_to(images_dir.resolve()):
                return jsonify({'error': 'Доступ запрещён'}), 403

            if not image_path.exists():
                return jsonify({'error': f'Изображение не найдено: {image_path}'}), 404

            label_path = labels_dir / f"{Path(image_name).stem}.txt"
            annotations = load_yolo_annotations(label_path)

            if len(annotations) == 0:
                # Если нет аннотаций, вернуть оригинальное изображение
                return send_file(str(image_path), mimetype='image/jpeg')

            # Создать визуализацию в памяти
            # Загрузить изображение
            img = cv2.imread(str(image_path))
            if img is None:
                return jsonify({'error': f'Не удалось загрузить изображение: {image_path}'}), 500

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

                # Подпись с классом
                label = f"Class {ann.class_id}"
                if app.config['CLASS_NAMES'] and ann.class_id in app.config['CLASS_NAMES']:
                    label = app.config['CLASS_NAMES'][ann.class_id]

                # Фон для текста
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)

                # Текст
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Конвертировать в JPEG в памяти
            success, buffer = cv2.imencode('.jpg', img)
            if not success:
                return jsonify({'error': 'Не удалось закодировать изображение'}), 500

            # Вернуть как response
            response = io.BytesIO(buffer.tobytes())
            response.seek(0)
            return send_file(response, mimetype='image/jpeg')

        except Exception as e:
            return jsonify({'error': f'Ошибка создания визуализации: {str(e)}'}), 500

    @app.route('/api/cache/clear', methods=['POST'])
    def api_clear_cache():
        """API: Очистить кэш изображений"""
        try:
            if hasattr(app, 'image_cache'):
                app.image_cache.clear()
            return jsonify({'message': 'Кэш очищен'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app


def run_dataset_viewer(images_dir: Union[str, Path],
                      labels_dir: Union[str, Path],
                      host: str = '127.0.0.1',
                      port: int = 5000,
                      debug: bool = False,
                      class_names: Optional[Dict[int, str]] = None):
    """
    Запустить веб-интерфейс для просмотра датасета

    Args:
        images_dir: Директория с изображениями
        labels_dir: Директория с аннотациями
        host: Хост для запуска сервера
        port: Порт для запуска сервера
        debug: Режим отладки
        class_names: Словарь {class_id: class_name}
    """
    app = create_dataset_viewer_app(images_dir, labels_dir, class_names)

    print(f"Запуск Dataset Viewer...")
    print(f"Изображения: {images_dir}")
    print(f"Аннотации: {labels_dir}")
    print(f"Открыть в браузере: http://{host}:{port}")
    print("Нажмите Ctrl+C для остановки сервера")

    app.run(host=host, port=port, debug=debug)
