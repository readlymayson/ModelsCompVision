#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

def check_images():
    images_dir = "data/images"
    if not os.path.exists(images_dir):
        print(f"Директория {images_dir} не существует")
        return

    files = os.listdir(images_dir)
    print(f"Всего файлов: {len(files)}")

    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    print(f"JPG файлов: {len(jpg_files)}")

    # Проверим первые 5 файлов
    print("Первые 5 файлов:")
    for f in jpg_files[:5]:
        filepath = os.path.join(images_dir, f)
        try:
            # Попробуем прочитать начало файла
            with open(filepath, 'rb') as file:
                header = file.read(10)
                print(f"  {f}: {header[:4].hex()} (размер: {os.path.getsize(filepath)} байт)")
        except Exception as e:
            print(f"  {f}: ОШИБКА - {e}")

    # Проверим на битые файлы
    print("\nПроверка на битые файлы:")
    bad_files = 0
    for f in jpg_files[:20]:  # Проверим первые 20 файлов
        filepath = os.path.join(images_dir, f)
        try:
            with open(filepath, 'rb') as file:
                header = file.read(4)
                if header != b'\xff\xd8\xff\xe0' and header != b'\xff\xd8\xff\xe1':
                    print(f"  {f}: НЕ JPG заголовок - {header.hex()}")
                    bad_files += 1
        except Exception as e:
            print(f"  {f}: ОШИБКА чтения - {e}")
            bad_files += 1

    print(f"\nНайдено подозрительных файлов: {bad_files}")

if __name__ == "__main__":
    check_images()
