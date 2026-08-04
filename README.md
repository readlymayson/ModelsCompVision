# Video2Dataset

Инструмент для подготовки датасетов компьютерного зрения из видео под YOLOv8.

## Возможности

- **Разбиение видео на кадры**: Извлечение кадров из видео в формате JPG с настраиваемой частотой
- **Автоматическая разметка**: Использование предобученной модели YOLOv8 для автоматической генерации аннотаций
- **Аугментации**: Генерация дополнительных данных через трансформации изображений и аннотаций
- **Валидация**: Проверка корректности аннотаций в формате YOLO
- **Подготовка к обучению**: Генерация конфигурационных файлов для обучения YOLOv8
- **Фильтрация пустых аннотаций**: Автоматическое перемещение изображений без аннотаций в отдельную папку

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

### 1. Разбить видео на кадры

```bash
python -m video2dataset.cli split-video data/videos/sample.mp4 --output data/images --fps 1.0
```

### 2. Автоматическая разметка

```bash
python -m video2dataset.cli auto-annotate --images data/images --labels data/labels
```

### 3. Аугментация датасета

```bash
python -m video2dataset.cli augment --images data/images --labels data/labels --output-images data/augmented/images --output-labels data/augmented/labels --num-augs 3
```

### 4. Валидация аннотаций

```bash
python -m video2dataset.cli validate-labels --images data/images --labels data/labels
```

### 5. Подготовка к обучению YOLOv8

```bash
python -m video2dataset.cli make-yolo-config --images data/images --labels data/labels --output data.yaml
```

### 6. Фильтрация пустых аннотаций

Для очистки датасета от изображений без аннотаций используйте скрипт `filter_empty_annotations.py`:

```bash
python filter_empty_annotations.py
```

Скрипт автоматически:
- Найдет все файлы аннотаций без содержимого в папке `data/labels`
- Переместит соответствующие изображения из `data/images` в `data/неаннотированно/images`
- Переместит пустые файлы аннотаций в `data/неаннотированно/labels`

Это позволяет отделить неаннотированные изображения от размеченного датасета для дальнейшей обработки.

### 7. Просмотр датасета

Для визуального просмотра и анализа датасета используйте встроенный веб-интерфейс:

```bash
python -m video2dataset.cli view-dataset --images data/images --labels data/labels --port 5000
```

Веб-интерфейс предоставляет:
- Галерею изображений с превью
- Просмотр отдельных изображений с аннотациями
- Статистику датасета (распределение классов, количество изображений)
- Фильтрацию по классам и поиск по имени файла

## Структура проекта

```
video2dataset/
├── __init__.py
├── config.py              # Глобальные настройки
├── cli.py                 # Командная строка
├── video_splitter.py      # Разбиение видео на кадры
├── annotation_formats.py  # Работа с аннотациями YOLO
├── auto_annotator.py      # Автоматическая разметка
├── augmentations.py       # Аугментации изображений
├── yolo_integration.py    # Интеграция с YOLOv8
└── dataset_viewer.py      # Веб-интерфейс для просмотра датасета

data/
├── videos/                # Исходные видео
├── images/                # Извлеченные кадры
├── labels/                # Аннотации YOLO
├── augmented/             # Аугментированные данные
└── previews/              # Визуальные превью
```

## Команды CLI

### split-video
Разбивает видео на кадры JPG.

Параметры:
- `--fps FLOAT`: Частота извлечения кадров (кадры в секунду)
- `--every-n INT`: Извлекать каждый N-й кадр
- `--output DIR`: Директория для сохранения изображений
- `--resize WIDTH HEIGHT`: Изменить размер изображений

### auto-annotate
Автоматически разметить изображения с помощью YOLOv8.

Параметры:
- `--images DIR`: Директория с изображениями
- `--labels DIR`: Директория для сохранения аннотаций
- `--model PATH`: Путь к модели YOLO8 (по умолчанию yolov8n.pt)
- `--conf FLOAT`: Порог уверенности (по умолчанию 0.25)
- `--iou FLOAT`: Порог IoU для NMS (по умолчанию 0.45)
- `--mode`: Режим работы (fill_missing/overwrite/backup)


### augment
Создать аугментированные версии датасета.

Параметры:
- `--images DIR`: Исходные изображения
- `--labels DIR`: Исходные аннотации
- `--output-images DIR`: Выходные изображения
- `--output-labels DIR`: Выходные аннотации
- `--num-augs INT`: Количество аугментаций на изображение
- `--transforms LIST`: Список трансформаций для применения

### validate-labels
Проверить корректность аннотаций.

Параметры:
- `--images DIR`: Директория с изображениями
- `--labels DIR`: Директория с аннотациями
- `--fix`: Автоматически исправить найденные ошибки

### make-yolo-config
Создать конфигурационный файл data.yaml для обучения YOLOv8.

Параметры:
- `--images DIR`: Директория с изображениями
- `--labels DIR`: Директория с аннотациями
- `--classes LIST`: Список классов (если не указан, будет определен автоматически)
- `--output PATH`: Путь для сохранения data.yaml

### view-dataset
Запустить веб-интерфейс для просмотра и анализа датасета.

Параметры:
- `--images DIR`: Директория с изображениями
- `--labels DIR`: Директория с аннотациями
- `--host HOST`: Хост для запуска сервера (по умолчанию: 127.0.0.1)
- `--port PORT`: Порт для запуска сервера (по умолчанию: 5000)
- `--debug`: Включить режим отладки
- `--classes LIST`: Имена классов для отображения

## Примеры использования

### Полный пайплайн подготовки датасета

```bash
# 1. Разбить видео на кадры
python -m video2dataset.cli split-video data/videos/ --output data/images --fps 4.0

# 2. Автоматическая разметка (только для отсутствующих аннотаций)
python -m video2dataset.cli auto-annotate --images data/images --labels data/labels --mode fill_missing

python -m video2dataset.cli auto-annotate --images data/images --labels data/labels --mode fill_missing --model models\MediumVideocartFP16.pt --conf 0.7

# 3. (Вариант с аугментацией) Аугментация для увеличения датасета
python -m video2dataset.cli augment --images data/images --labels data/labels --output-images data/augmented/images --output-labels data/augmented/labels --num-augs 5

# 4. (Вариант с аугментацией) Валидация финального датасета
python -m video2dataset.cli validate-labels --images data/augmented/images --labels data/augmented/labels

# 5. (Вариант с аугментацией) Создание конфига для обучения
python -m video2dataset.cli make-yolo-config --images data/augmented/images --labels data/augmented/labels --output data.yaml

---

# 3. (Вариант без аугментации) Валидация исходных аннотаций
python -m video2dataset.cli validate-labels --images data/images --labels data/labels

# 4. (Вариант без аугментации) Создание конфига для обучения
python -m video2dataset.cli make-yolo-config --images dataset_annoyed\images --labels dataset_annoyed\labels --output data.yaml

# 6. Обучение модели (с помощью ultralytics)
yolo train data=data.yaml model=yolov8n.pt epochs=10 imgsz=420

yolo train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640

yolo train data=data.yaml model=yolov8n.pt epochs=100 imgsz=1024

# 7. Просмотр датасета в веб-интерфейсе
python -m video2dataset.cli view-dataset --images data/images --labels data/labels --port 8080

## Автоматизированные скрипты

### Подготовка датасета

```bash
# Полная подготовка датасета (разделение + data.yaml)
python prepare_dataset.py

# С кастомными параметрами
python prepare_dataset.py --train-split 0.8 --val-split 0.15 --test-split 0.05 --force-recreate

# Просмотр всех опций
python prepare_dataset.py --help
```

### Обучение YOLO

```bash
# Обучение с подготовленным датасетом
train_yolo.bat

# Полная подготовка и обучение
train_complete.bat
```

### Управление данными

```bash
# Фильтрация пустых аннотаций
python filter_empty_annotations.py

# Перемещение данных в dataset_annoyed
python move_to_dataset_annoyed.py
```

## Требования

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- Albumentations
- NumPy, Pillow, PyYAML, tqdm, Click

## Лицензия

MIT License
