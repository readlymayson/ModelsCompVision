# -*- coding: utf-8 -*-
"""
Модуль для разбиения видео на кадры JPG
"""

import cv2
import logging
from pathlib import Path
from typing import Optional, Union, Tuple
import tqdm

from .config import DEFAULT_IMAGE_FORMAT, DEFAULT_IMAGE_QUALITY, DEFAULT_ENCODING

logger = logging.getLogger(__name__)


class VideoSplitter:
    """Класс для разбиения видео на кадры"""

    def __init__(self, output_format: str = DEFAULT_IMAGE_FORMAT, quality: int = DEFAULT_IMAGE_QUALITY):
        """
        Инициализация сплиттера видео

        Args:
            output_format: Формат выходных изображений ('jpg', 'png', etc.)
            quality: Качество JPEG (0-100)
        """
        self.output_format = output_format.lower()
        self.quality = quality

        # Настройки для сохранения изображений
        if self.output_format == 'jpg':
            self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        else:
            self.encode_param = []

    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Получить информацию о видео

        Args:
            video_path: Путь к видео файлу

        Returns:
            Словарь с информацией о видео
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Видео файл не найден: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            return {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration,
                'path': video_path
            }
        finally:
            cap.release()

    def calculate_frame_indices(self, video_info: dict, fps: Optional[float] = None,
                               every_n: Optional[int] = None) -> list:
        """
        Рассчитать индексы кадров для извлечения

        Args:
            video_info: Информация о видео
            fps: Желаемая частота кадров в секунду (например, 1.0 = 1 кадр в секунду)
            every_n: Извлекать каждый N-й кадр

        Returns:
            Список индексов кадров
        """
        total_frames = video_info['total_frames']
        video_fps = video_info['fps']
        duration = video_info['duration']

        if every_n is not None:
            # Извлекать каждый N-й кадр
            frame_step = every_n
            indices = list(range(0, total_frames, frame_step))
        elif fps is not None and fps > 0:
            # Извлекать с заданной частотой кадров в секунду
            # fps=1.0 означает 1 кадр в секунду, fps=2.0 означает 2 кадра в секунду и т.д.
            if fps >= video_fps:
                # Если запрошено больше кадров, чем есть в видео, берем все кадры
                indices = list(range(0, total_frames, 1))
            else:
                # Рассчитываем шаг: для fps=1.0 и video_fps=30 -> шаг = 30 (каждую секунду)
                frame_step = max(1, int(video_fps / fps))
                indices = list(range(0, total_frames, frame_step))
        else:
            # По умолчанию каждый кадр
            indices = list(range(0, total_frames, 1))

        logger.info(f"Видео: {total_frames} кадров, {video_fps:.1f} FPS, длительность: {duration:.1f} сек")
        logger.info(f"Будет извлечено {len(indices)} кадров (fps={fps}, every_n={every_n})")

        return indices

    def extract_frames(self, video_path: Union[str, Path], output_dir: Union[str, Path],
                      fps: Optional[float] = None, every_n: Optional[int] = None,
                      resize: Optional[Tuple[int, int]] = None,
                      prefix: Optional[str] = None) -> list:
        """
        Извлечь кадры из видео

        Args:
            video_path: Путь к видео
            output_dir: Директория для сохранения кадров
            fps: Частота извлечения кадров (кадры в секунду)
            every_n: Извлекать каждый N-й кадр
            resize: Кортеж (width, height) для изменения размера
            prefix: Префикс для имен файлов

        Returns:
            Список путей к сохраненным изображениям
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Получить информацию о видео
        video_info = self.get_video_info(video_path)
        logger.info(f"Видео: {video_info['width']}x{video_info['height']}, "
                   f"{video_info['total_frames']} кадров, {video_info['fps']:.2f} FPS")

        # Рассчитать индексы кадров
        frame_indices = self.calculate_frame_indices(video_info, fps, every_n)
        logger.info(f"Будет извлечено {len(frame_indices)} кадров")

        # Открыть видео
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

        # Генерировать базовое имя файла
        if prefix:
            base_name = f"{prefix}_{video_path.stem}"
        else:
            base_name = video_path.stem

        saved_paths = []

        try:
            # Прогресс-бар
            with tqdm.tqdm(total=len(frame_indices), desc=f"Обработка {video_path.name}") as pbar:
                for i, frame_idx in enumerate(frame_indices):
                    # Перейти к нужному кадру
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if not ret:
                        logger.warning(f"Не удалось прочитать кадр {frame_idx}")
                        continue

                    # Изменить размер если нужно
                    if resize:
                        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)

                    # Сгенерировать имя файла
                    frame_name = f"{i:06d}"
                    output_path = output_dir / f"{base_name}_frame_{frame_name}.{self.output_format}"

                    # Сохранить изображение
                    success = cv2.imwrite(str(output_path), frame, self.encode_param)
                    if success:
                        saved_paths.append(output_path)
                    else:
                        logger.error(f"Не удалось сохранить кадр: {output_path}")

                    pbar.update(1)

        finally:
            cap.release()

        logger.info(f"Извлечено {len(saved_paths)} кадров из {video_path.name}")
        return saved_paths

    def process_video_directory(self, input_dir: Union[str, Path], output_dir: Union[str, Path],
                               fps: Optional[float] = None, every_n: Optional[int] = None,
                               resize: Optional[Tuple[int, int]] = None,
                               video_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')) -> dict:
        """
        Обработать директорию с видео файлами

        Args:
            input_dir: Директория с видео
            output_dir: Директория для сохранения кадров
            fps: Частота извлечения кадров
            every_n: Извлекать каждый N-й кадр
            resize: Размер для изменения изображений
            video_extensions: Расширения видео файлов

        Returns:
            Словарь с результатами обработки
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {input_dir}")

        # Найти все видео файлы
        video_files = []
        for ext in video_extensions:
            found_lower = list(input_dir.glob(f"**/*{ext}"))
            found_upper = list(input_dir.glob(f"**/*{ext.upper()}"))
            video_files.extend(found_lower)
            video_files.extend(found_upper)

        # Удалить дубликаты
        video_files = list(set(video_files))

        if not video_files:
            logger.warning(f"Видео файлы не найдены в директории: {input_dir}")
            return {'total_videos': 0, 'total_frames': 0, 'videos': []}

        logger.info(f"Найдено {len(video_files)} видео файлов: {[str(f) for f in video_files]}")

        total_frames = 0
        results = []

        for video_path in video_files:
            try:
                # Создать поддиректорию для каждого видео
                video_output_dir = output_dir / video_path.stem
                video_output_dir.mkdir(parents=True, exist_ok=True)

                # Извлечь кадры
                saved_paths = self.extract_frames(
                    video_path, video_output_dir, fps, every_n, resize
                )

                results.append({
                    'video_path': video_path,
                    'frames_extracted': len(saved_paths),
                    'output_dir': video_output_dir
                })

                total_frames += len(saved_paths)

            except Exception as e:
                logger.error(f"Ошибка обработки видео {video_path}: {e}")
                results.append({
                    'video_path': video_path,
                    'error': str(e),
                    'frames_extracted': 0
                })

        return {
            'total_videos': len(video_files),
            'total_frames': total_frames,
            'videos': results
        }


def split_video(video_path: Union[str, Path], output_dir: Union[str, Path],
               fps: Optional[float] = None, every_n: Optional[int] = None,
               resize: Optional[Tuple[int, int]] = None,
               output_format: str = DEFAULT_IMAGE_FORMAT,
               quality: int = DEFAULT_IMAGE_QUALITY) -> list:
    """
    Удобная функция для разбиения одного видео на кадры

    Args:
        video_path: Путь к видео
        output_dir: Директория для сохранения
        fps: Частота кадров
        every_n: Каждый N-й кадр
        resize: Размер изображений
        output_format: Формат изображений
        quality: Качество JPEG

    Returns:
        Список путей к сохраненным изображениям
    """
    splitter = VideoSplitter(output_format, quality)
    return splitter.extract_frames(video_path, output_dir, fps, every_n, resize)


def split_video_directory(input_dir: Union[str, Path], output_dir: Union[str, Path],
                         fps: Optional[float] = None, every_n: Optional[int] = None,
                         resize: Optional[Tuple[int, int]] = None,
                         output_format: str = DEFAULT_IMAGE_FORMAT,
                         quality: int = DEFAULT_IMAGE_QUALITY) -> dict:
    """
    Удобная функция для обработки директории с видео

    Args:
        input_dir: Директория с видео
        output_dir: Директория для сохранения
        fps: Частота кадров
        every_n: Каждый N-й кадр
        resize: Размер изображений
        output_format: Формат изображений
        quality: Качество JPEG

    Returns:
        Результаты обработки
    """
    splitter = VideoSplitter(output_format, quality)
    return splitter.process_video_directory(input_dir, output_dir, fps, every_n, resize)
