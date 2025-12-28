#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой скрипт для запуска Video2Dataset CLI
"""

import sys
import os

# Добавить текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video2dataset.cli import main

if __name__ == '__main__':
    sys.exit(main())
