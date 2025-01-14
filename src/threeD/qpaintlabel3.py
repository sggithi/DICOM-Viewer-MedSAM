import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt

class QPaintLabel3(QLabel):

    mpsignal = pyqtSignal(str)
    crosshairDrawingNeeded = pyqtSignal()
    updateNeeded = pyqtSignal()
    bounding_box_resized = pyqtSignal(QRectF)

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
        
        self.pos_xy = []
        self.crosscenter = [0, 0]
        self.mouseclicked = None
        self.sliceclick = False
    
        self.type = 'general'
        self.slice_loc = [0, 0, 0]
        self.slice_loc_restore = [0, 0, 0]
        self.mousein = False
        
        self.setMouseTracking(True)
        self.drag_start = None
        self.drag_end = None

        self.toggleBoundingBoxEnabled = False
        self.toggleSlicerEnabled = False

        self.pen_color = Qt.red  # 기본 펜 색상
        self.crosshairDrawingNeeded.connect(self.update)

        ## MedSAM을 위한 변수
        self.bounding_box = None
        self.image_loaded = False 

    # 마우스 버튼을 눌렀을 때 이벤트 처리
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self.bounding_box is None or not self.bounding_box.handleAt(event.pos()):
                self.drag_start = event.pos()
                self.drag_end = event.pos()
            if self.bounding_box is not None:
                self.bounding_box.mousePressEvent(event)
            self.update()

    # 마우스를 움직였을 때 이벤트 처리
    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        
        if self.parentReference.toggleBoundingBoxEnabled and event.buttons() & Qt.LeftButton:
            if self.bounding_box is not None and self.bounding_box.interactiveResize:
                self.bounding_box.mouseMoveEvent(event)
            else:
                self.drag_end = event.pos()
            self.update()

        if self.parentReference.toggleSlicerEnabled and event.buttons() & Qt.LeftButton:
            wl_adjustment = self.drag_end.x() - self.drag_start.x()
            ww_adjustment = self.drag_end.y() - self.drag_start.y()
            self.parentReference.windowLevel += wl_adjustment
            self.parentReference.windowWidth = max(1, self.parentReference.windowWidth + ww_adjustment)
            self.parentReference.updateimg()
            self.drag_start = event.pos()

    # 마우스 버튼을 뗐을 때 이벤트 처리
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_end = event.pos()
            if self.parentReference.toggleBoundingBoxEnabled:
                if self.bounding_box is None or not self.bounding_box.interactiveResize:
                    rect = QRectF(self.drag_start, self.drag_end).normalized()
                    self.bounding_box = ResizableRectItem(rect)
                    self.bounding_box.rectResized.connect(self.handle_bounding_box_resized)

                    # 다른 평면에 대해 ResizableRectItem 인스턴스 생성
                    if self.type == 'axial':
                        sagittal_rect = self.parentReference.map_rect_to_plane(rect, 'axial', 'sagittal')
                        coronal_rect = self.parentReference.map_rect_to_plane(rect, 'axial', 'coronal')
                        self.parentReference.imgLabel_2.bounding_box = ResizableRectItem(sagittal_rect)
                        self.parentReference.imgLabel_3.bounding_box = ResizableRectItem(coronal_rect)
                    elif self.type == 'sagittal':
                        axial_rect = self.parentReference.map_rect_to_plane(rect, 'sagittal', 'axial')
                        coronal_rect = self.parentReference.map_rect_to_plane(rect, 'sagittal', 'coronal')
                        self.parentReference.imgLabel_1.bounding_box = ResizableRectItem(axial_rect)
                        self.parentReference.imgLabel_3.bounding_box = ResizableRectItem(coronal_rect)
                    elif self.type == 'coronal':
                        axial_rect = self.parentReference.map_rect_to_plane(rect, 'coronal', 'axial')
                        sagittal_rect = self.parentReference.map_rect_to_plane(rect, 'coronal', 'sagittal')
                        self.parentReference.imgLabel_1.bounding_box = ResizableRectItem(axial_rect)
                        self.parentReference.imgLabel_2.bounding_box = ResizableRectItem(sagittal_rect)

                    self.parentReference.imgLabel_1.bounding_box.rectResized.connect(self.parentReference.update_bounding_boxes)
                    self.parentReference.imgLabel_2.bounding_box.rectResized.connect(self.parentReference.update_bounding_boxes)
                    self.parentReference.imgLabel_3.bounding_box.rectResized.connect(self.parentReference.update_bounding_boxes)
                    
                    # 즉시 다른 평면 업데이트
                    self.parentReference.imgLabel_1.update()
                    self.parentReference.imgLabel_2.update()
                    self.parentReference.imgLabel_3.update()
                
                else:
                    self.bounding_box.mouseReleaseEvent(event)
                self.update()

                self.drag_start = None
                self.drag_end = None

    # 바운딩 박스가 크기 조정되었을 때 호출되는 함수
    def handle_bounding_box_resized(self, rect):
        self.bounding_box_resized.emit(rect)  # 업데이트된 사각형과 함께 신호를 발행

    # 마우스가 라벨 영역을 벗어났을 때 이벤트 처리
    def leaveEvent(self, event):
        self.slice_loc = self.slice_loc_restore
        self.update()
        
    # 이미지를 표시하는 함수
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

    # 페인트 이벤트 처리
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

        # 바운딩 박스를 그리기
        if self.parentReference.toggleBoundingBoxEnabled and self.bounding_box is not None and self.image_loaded:
            painter.setPen(QPen(Qt.red, 3))
            painter.drawRect(self.bounding_box.rect)

            # 핸들 그리기
            painter.setBrush(QBrush(Qt.red))
            for handle_rect in self.bounding_box.handles.values():
                painter.drawRect(handle_rect)

# 크기 조정 가능한 사각형 항목 클래스
class ResizableRectItem(QObject):
    rectResized = pyqtSignal(QRectF)

    def __init__(self, rect, parent=None):
        super().__init__(parent)
        self.rect = rect
        self.handles = {}
        self.handleWidth = 15  # 핸들의 너비
        self.handleHeight = 15  # 핸들의 높이
        self.updateHandlesPositions()
        self.interactiveResize = False
        self.currentHandle = None

    # 특정 위치에 핸들이 있는지 확인하는 함수
    def handleAt(self, point):
        for handle, rect in self.handles.items():
            if rect.contains(point):
                return handle
        return None

    # 핸들의 위치를 업데이트하는 함수
    def updateHandlesPositions(self):
        w = self.handleWidth
        h = self.handleHeight
        rect = self.rect

        # 핸들의 위치를 계산
        topMiddle = QPointF(rect.left() + (rect.width() - w) / 2, rect.top() - h / 2)
        bottomMiddle = QPointF(rect.left() + (rect.width() - w) / 2, rect.bottom() + h / 2 - h)
        leftMiddle = QPointF(rect.left() - w / 2, rect.top() + (rect.height() - h) / 2)
        rightMiddle = QPointF(rect.right() + w / 2 - w, rect.top() + (rect.height() - h) / 2)

        # 각 핸들에 대해 QRectF 객체 생성
        self.handles['top'] = QRectF(topMiddle, QSizeF(w, h))
        self.handles['bottom'] = QRectF(bottomMiddle, QSizeF(w, h))
        self.handles['left'] = QRectF(leftMiddle, QSizeF(w, h))
        self.handles['right'] = QRectF(rightMiddle, QSizeF(w, h))

    # 마우스 버튼을 눌렀을 때 이벤트 처리
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.currentHandle = self.handleAt(event.pos())
            self.interactiveResize = self.currentHandle is not None
            if self.interactiveResize:
                event.accept()

    # 마우스를 움직였을 때 이벤트 처리
    def mouseMoveEvent(self, event):
        if self.interactiveResize:
            self.resizeItem(event.pos())
            event.accept()

    # 마우스 버튼을 뗐을 때 이벤트 처리
    def mouseReleaseEvent(self, event):
        self.interactiveResize = False
        self.currentHandle = None

    # 항목의 크기를 조정하는 함수
    def resizeItem(self, pos):
        if 'top' == self.currentHandle:
            newTop = min(max(pos.y(), 0), 511 - self.rect.height())
            self.rect.setTop(newTop)
        elif 'bottom' == self.currentHandle:
            newBottom = min(max(pos.y(), self.rect.top() + 1), 511)
            self.rect.setBottom(newBottom)
        elif 'left' == self.currentHandle:
            newLeft = min(max(pos.x(), 0), 511 - self.rect.width())
            self.rect.setLeft(newLeft)
        elif 'right' == self.currentHandle:
            newRight = min(max(pos.x(), self.rect.left() + 1), 511)
            self.rect.setRight(newRight)
        self.updateHandlesPositions()
        self.rectResized.emit(self.rect)
