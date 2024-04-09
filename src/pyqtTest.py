import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QRect


class BoundingBoxWidget(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Bounding Box Example')
        self.setGeometry(100, 100, 600, 400)
        self.setMouseTracking(True)
        self.drag_start = None
        self.drag_end = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(255, 0, 0))  # Set color to red
        if self.drag_start and self.drag_end:
            rect = QRect(self.drag_start, self.drag_end).normalized()
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        print(event.type)
        print(event)
        if event.button() == Qt.LeftButton:
            self.drag_start = event.pos()
            self.drag_end = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.drag_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_end = event.pos()
            print(self.drag_start, self.drag_end)
            self.update()
            self.drag_start = None
            self.drag_end = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BoundingBoxWidget()
    window.show()
    sys.exit(app.exec_())

