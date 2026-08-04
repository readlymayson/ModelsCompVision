#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для проверки CUDA доступности
"""

import torch

print('=== PyTorch CUDA Information ===')
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version in torch:', getattr(torch.version, 'cuda', 'None'))
print('cuDNN enabled:', torch.backends.cudnn.enabled if hasattr(torch.backends, 'cudnn') else False)

if torch.backends.cudnn.enabled:
    try:
        print('cuDNN version:', torch.backends.cudnn.version())
    except:
        print('cuDNN version: Not available')

print('Device count:', torch.cuda.device_count())

if torch.cuda.is_available():
    print('Current device:', torch.cuda.current_device())
    print('Device name:', torch.cuda.get_device_name())
    print('Device capability:', torch.cuda.get_device_capability())
    print('Memory allocated:', f"{torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print('Memory cached:', f"{torch.cuda.memory_reserved()/1024**3:.2f} GB")
else:
    print('CUDA devices: Not available')

# Проверка ultralytics
try:
    import ultralytics
    print('\n=== Ultralytics Information ===')
    print('Ultralytics version:', ultralytics.__version__)
    from ultralytics import YOLO
    print('YOLO import: OK')
except ImportError as e:
    print('Ultralytics import error:', e)





