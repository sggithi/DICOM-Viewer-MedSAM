import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import numpy as np
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class QPaintLabel3(QLabel):

    mpsignal = pyqtSignal(str)
    crosshairDrawingNeeded = pyqtSignal()

    def __init__(self, parent):
        super(QLabel, self).__init__(parent)
        self.parentReference = parent
        self.setMinimumSize(1, 1)
        self.image = None
        self.processedImage = None
        self.imgr, self.imgc = None, None
        self.imgpos_x, self.imgpos_y = None, None
        self.pos_x = 20
        self.pos_y = 20
        self.imgr, self.imgc = None, None
        # 遇到list就停，圖上的顯示白色只是幌子
        self.pos_xy = []
        # 十字的中心點！每個QLabel指定不同中心點，這樣可以用一樣的paintevent function
        self.crosscenter = [0, 0]
        self.mouseclicked = None
        self.sliceclick = False
        # 決定用哪種paintEvent的type, general代表一般的
        self.type = 'general'
        self.slice_loc = [0, 0, 0]
        self.slice_loc_restore = [0, 0, 0]
        self.mousein = False
        
        self.setMouseTracking(True)
        self.drag_start = None
        self.drag_end = None

        self.toggleBoundingBoxEnabled = False
        self.toggleSlicerEnabled = False
        self.bounding_box = None

        self.pen_color = Qt.red  # Default pen color
        self.crosshairDrawingNeeded.connect(self.update)


    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        
        # MARK: BoundingBox
        # Update the drag_end position for both bounding box and slicer functionality
        self.drag_end = event.pos()
        self.update()

        if self.parentReference.toggleSlicerEnabled and event.buttons() & Qt.LeftButton:
            # Adjust WW and WL for slicer functionality
            wl_adjustment = self.drag_end.x() - self.drag_start.x()
            ww_adjustment = self.drag_end.y() - self.drag_start.y()
            self.parent().windowLevel += wl_adjustment
            self.parent().windowWidth = max(1, self.parent().windowWidth + ww_adjustment)
            self.parent().updateimg()
            self.drag_start = self.drag_end




    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_end = event.pos()
            if self.parentReference.toggleBoundingBoxEnabled:
                # Store the bounding box coordinates
                self.bounding_box = QRect(self.drag_start, self.drag_end).normalized()
                # Print the bounding box coordinates
                print(np.array([self.drag_start.x(), self.drag_start.y(), self.drag_end.x(), self.drag_start.y()]))
            self.update()
            self.drag_start = None
            self.drag_end = None

    def leaveEvent(self, event):
        self.slice_loc = self.slice_loc_restore
        
        self.update()
        
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drag_start = event.pos()
            self.drag_end = event.pos()
            self.update()

    def display_image(self, window=1):
        self.imgr, self.imgc = self.processedImage.shape[0:2]
        qformat = QImage.Format_Indexed8
        if len(self.processedImage.shape) == 3:  # rows[0], cols[1], channels[2]
            if (self.processedImage.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.processedImage, self.processedImage.shape[1], self.processedImage.shape[0],
                     self.processedImage.strides[0], qformat)
        img = img.rgbSwapped()
        w, h = self.width(), self.height()
        if window == 1:
            self.setScaledContents(True)
            backlash = self.lineWidth() * 2
            self.setPixmap(QPixmap.fromImage(img).scaled(w - backlash, h - backlash, Qt.IgnoreAspectRatio))
            self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
            
        loc = QFont()
        loc.setPixelSize(10)
        loc.setBold(True)
        loc.setItalic(True)
        loc.setPointSize(15)
        
        if self.pixmap():
            pixmap = self.pixmap()
            painter = QPainter(self)
            painter.drawPixmap(self.rect(), pixmap)

            # Draw the cross-hair lines
            if self.type == 'axial':
                # Draw vertical line
                painter.setPen(QPen(Qt.red, 3))
                painter.drawLine(self.crosscenter[0], 0, self.crosscenter[0], self.height())
                # Draw horizontal line
                painter.setPen(QPen(Qt.cyan, 3))
                painter.drawLine(0, self.crosscenter[1], self.width(), self.crosscenter[1])
                # Draw center point
                painter.setPen(QPen(Qt.yellow, 3))
                painter.drawPoint(self.crosscenter[0], self.crosscenter[1])
            elif self.type == 'sagittal':
                # Draw vertical line
                painter.setPen(QPen(Qt.cyan, 3))
                painter.drawLine(self.crosscenter[0], 0, self.crosscenter[0], self.height())
                # Draw horizontal line
                painter.setPen(QPen(Qt.yellow, 3))
                painter.drawLine(0, self.crosscenter[1], self.width(), self.crosscenter[1])
                # Draw center point
                painter.setPen(QPen(Qt.red, 3))
                painter.drawPoint(self.crosscenter[0], self.crosscenter[1])
            elif self.type == 'coronal':
                # Draw vertical line
                painter.setPen(QPen(Qt.red, 3))
                painter.drawLine(self.crosscenter[0], 0, self.crosscenter[0], self.height())
                # Draw horizontal line
                painter.setPen(QPen(Qt.yellow, 3))
                painter.drawLine(0, self.crosscenter[1], self.width(), self.crosscenter[1])
                # Draw center point
                painter.setPen(QPen(Qt.cyan, 3))
                painter.drawPoint(self.crosscenter[0], self.crosscenter[1])
            else:
                pass

                self.crosshairDrawingNeeded.emit()
        # Draw the bounding box if it's enabled
        if self.parentReference.toggleBoundingBoxEnabled and self.drag_start and self.drag_end:
            rect = QRect(self.drag_start, self.drag_end).normalized()
            painter.drawRect(rect)
        if self.parentReference.toggleBoundingBoxEnabled and self.bounding_box is not None:
                painter.setPen(QPen(Qt.red, 3))
                painter.drawRect(self.bounding_box)

            

def linear_convert(img):
    convert_scale = 255.0 / (np.max(img) - np.min(img))
    converted_img = convert_scale*img-(convert_scale*np.min(img))
    return converted_img
