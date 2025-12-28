# -*- coding: utf-8 -*-
"""Тест множественного наследования для ResizableRectItem"""

from PyQt5.QtWidgets import QGraphicsRectItem
from PyQt5.QtCore import QObject, pyqtSignal

class TestResizableRectItem(QGraphicsRectItem, QObject):
    """Тестовый класс с множественным наследованием"""

    geometry_changed = pyqtSignal(int, float, float, float, float)

    def __init__(self, x, y, width, height, annotation_index, class_id):
        # Важно: сначала инициализировать QObject, затем QGraphicsRectItem
        QObject.__init__(self)
        QGraphicsRectItem.__init__(self, x, y, width, height)
        self.annotation_index = annotation_index
        self.class_id = class_id

if __name__ == '__main__':
    try:
        item = TestResizableRectItem(0, 0, 100, 100, 0, 0)
        print("Множественное наследование работает!")
        print(f"Тип: {type(item)}")
        print(f"Наследуется от QObject: {isinstance(item, QObject)}")
        print(f"Наследуется от QGraphicsRectItem: {isinstance(item, QGraphicsRectItem)}")
    except Exception as e:
        print(f"Ошибка: {e}")
