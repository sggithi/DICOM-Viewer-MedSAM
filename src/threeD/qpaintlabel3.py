import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class QPaintLabel3(QLabel):

    mpsignal = pyqtSignal(str)
    crosshairDrawingNeeded = pyqtSignal()
    updateNeeded = pyqtSignal()

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

        self.pen_color = Qt.red  # Default pen color
        self.crosshairDrawingNeeded.connect(self.update)

        ## for MedSAM
        self.dname = None
        self.pos_xyz_start = [] # shared by axial, sagital, cornoal
        self.pos_xyz_end = []
        self.box_origin = None
        self.draw = 0




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
            if self.parentReference.toggleBoundingBoxEnabled :
                # Print the bounding box coordinates
                # print(self.type, "type")
                if self.type == 'axial':
                    self.box_origin = 'axial'
                    self.pos_xyz_start = [self.drag_start.x(), self.drag_start.y(), 0]
                    self.pos_xyz_end = [self.drag_end.x(), self.drag_end.y(), 0]
                
                elif self.type == 'sagittal':
                    self.box_origin = 'sagittal'
                    self.pos_xyz_start = [0, self.drag_start.y(), self.drag_start.x()]
                    self.pos_xyz_end = [ self.drag_end.y(),self.drag_end.x(), 0]
               
                elif self.type == 'coronal':
                    self.box_origin = 'coronal'
                    self.pos_xyz_start = [self.drag_start.x(), 0, self.drag_start.y()]
                    self.pos_xyz_end = [self.drag_end.x(), 0, self.drag_end.y()]
                self.updateNeeded.emit()

            
            print(self.type, np.array([self.drag_start.x(), self.drag_start.y(), self.drag_end.x(), self.drag_start.y()]))
            self.draw = 1
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
            painter.setPen(QPen(Qt.red, 3))
            rect = QRect(self.drag_start, self.drag_end).normalized()
            painter.drawRect(rect)

        if self.parentReference.toggleBoundingBoxEnabled and self.draw == 1:
            print(self.box_origin, "origin")
            # if self.type == self.box_origin:
            #     painter.setPen(QPen(Qt.red, 3))
            #     rect = QRect(self.drag_start, self.drag_end).normalized()
            #     self.draw = 0
            #     painter.drawRect(rect)
            if self.type == 'axial':
                painter.setPen(QPen(Qt.red, 3))
                rect = QRect(QPoint(self.pos_xyz_start[0],self.pos_xyz_start[1]), QPoint(self.pos_xyz_end[0],self.pos_xyz_end[1])).normalized()
                painter.drawRect(rect)
                self.draw = 0
            elif self.type == 'sagittal':
                painter.setPen(QPen(Qt.red, 3))
                rect = QRect(QPoint(self.pos_xyz_start[2],self.pos_xyz_start[1]), QPoint(self.pos_xyz_end[2],self.pos_xyz_end[1])).normalized()
                painter.drawRect(rect)
                self.draw = 0
            elif self.type == 'coronal':
                painter.setPen(QPen(Qt.red, 3))
                self.draw = 0
                rect = QRect(QPoint(self.pos_xyz_start[0],self.pos_xyz_start[2]), QPoint(self.pos_xyz_end[0],self.pos_xyz_end[2])).normalized()
                painter.drawRect(rect)      


                

            

def linear_convert(img):
    convert_scale = 255.0 / (np.max(img) - np.min(img))
    converted_img = convert_scale*img-(convert_scale*np.min(img))
    return converted_img
