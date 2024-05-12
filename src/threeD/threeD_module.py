import sys
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import os
import cv2
import torch
import threeD.loaddicomfile as ldf
import numpy as np
from threeD.vol_view_module import C3dView
from threeD.qpaintlabel3 import QPaintLabel3
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets

from .inference_3D import medsam_lite_model, medsam_inference, resize_longest_side, pad_image, get_bbox
# with torch.no_grad():
#     img_256_tensor = torch.tensor(self.processedvoxel).float().permute(0, 3, 1, 2).to(device)
#     self.embedding = medsam_lite_model.image_encoder(img_256_tensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CthreeD(QDialog):
    def __init__(self):
        super().__init__()
        path = os.getcwd()
        os.chdir(path + '/threeD')
        self.directory = os.getcwd()
        loadUi('threeD_module.ui', self)
        self.setWindowTitle('3D Processing')
        self.UiComponents() 
        self.image = None
        self.voxel = None
        self.processedvoxel = None
        self.v1, self.v2, self.v3 = None, None, None
        self.volWindow = None
        self.dicomButton.clicked.connect(self.dicom_clicked)
        self.axial_hSlider.valueChanged.connect(self.updateimg)
        self.axial_vSlider.valueChanged.connect(self.updateimg)
        self.sagittal_hSlider.valueChanged.connect(self.updateimg)
        self.sagittal_vSlider.valueChanged.connect(self.updateimg)
        self.coronal_hSlider.valueChanged.connect(self.updateimg)
        self.coronal_vSlider.valueChanged.connect(self.updateimg)
        self.colormap = None

        self.volButton.clicked.connect(self.open_3dview)

        # self.w, self.h = self.imgLabel_1.width(), self.imgLabel_1.height()


        self.imgLabel_1.type = 'axial'
        self.imgLabel_2.type = 'sagittal'
        self.imgLabel_3.type = 'coronal'

        self.imgLabel_1.updateNeeded.connect(self.updateimg)
        self.imgLabel_2.updateNeeded.connect(self.updateimg)
        self.imgLabel_3.updateNeeded.connect(self.updateimg)

        self.axialGrid.setSpacing(1)
        self.saggitalGrid.setSpacing(1)
        self.coronalGrid.setSpacing(1)

        h = QSpacerItem(10, 10, QSizePolicy.Fixed, QSizePolicy.Fixed)
        v = QSpacerItem(10, 10, QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.axial_vBox.setSpacing(0)
        self.axial_vBox.insertSpacerItem(0, v)
        self.axial_vBox.insertSpacerItem(2, v)
        self.axial_hBox.setSpacing(0)
        self.axial_hBox.insertSpacerItem(0, h)
        self.axial_hBox.insertSpacerItem(2, h)
        self.saggital_vBox.setSpacing(0)
        self.saggital_vBox.insertSpacerItem(0, v)
        self.saggital_vBox.insertSpacerItem(2, v)
        self.saggital_hBox.setSpacing(0)
        self.saggital_hBox.insertSpacerItem(0, h)
        self.saggital_hBox.insertSpacerItem(2, h)
        self.coronal_vBox.setSpacing(0)
        self.coronal_vBox.insertSpacerItem(0, v)
        self.coronal_vBox.insertSpacerItem(2, v)
        self.coronal_hBox.setSpacing(0)
        self.coronal_hBox.insertSpacerItem(0, h)
        self.coronal_hBox.insertSpacerItem(2, h)

        self.colormap_hBox.insertStretch(2)
        self.colormap_hBox.insertSpacerItem(0, QSpacerItem(30, 0, QSizePolicy.Fixed,  QSizePolicy.Fixed))

        self.dcmInfo = None

        self.cross_recalc = True

        # for MedSAM 3D
        self.box_coordinates = [] # shared by sagital, axial, coronal
        self.imgLabel_1.setFixedSize(512, 512)
        self.imgLabel_2.setFixedSize(512, 512)
        self.imgLabel_3.setFixedSize(512, 512)
        

        # creating a push button 
        # Bounding Box & HUWindowing
        self.boundingBox.clicked.connect(self.toggle_bounding_box_functionality)
        self.boundingBox.setChecked(True)
        self.boundingBox.setStyleSheet('''
                                         QPushButton {
                                        background-color: #0E6AC2;
                                        border-style: solid;
                                        border-color: #A7B4BF;
                                        border-width: 1px;
                                        color: white;
                                        text-align: center;
                                        padding: 8px;
                                        font: bold 12px;
                                        min-width: 10em;
                                        border-radius: 15px;
                                        }
                                        ''')
        self.windowing.clicked.connect(self.toggle_slicer_functionality)
        self.windowing.setChecked(False)

        self.windowing.setEnabled(False)
        self.boundingBox.setEnabled(False)
        self.generateMask.setEnabled(False)

        # Generate Button 
        # Connect the clicked signal of the generateMask button to the generateEvent method
        self.generateMask.clicked.connect(self.generateEvent)

        self.imgLabel_1.bounding_box_resized.connect(self.update_bounding_boxes)
        self.imgLabel_2.bounding_box_resized.connect(self.update_bounding_boxes)
        self.imgLabel_3.bounding_box_resized.connect(self.update_bounding_boxes)

        self.segmentation_result = None
        self.origin_processedvoxel = None

    def UiComponents(self): 
        self.windowWidth = 400  # Default window width
        self.windowLevel = 40   # Default window level

        self.toggleSlicerEnabled = False
        self.toggleBoundingBoxEnabled = True

        # This spacer will push everything to the left of it to the left, and everything to the right of it to the right
        self.colormap_hBox.addStretch(1)

        self.wwlLabel = QLabel(self)
        self.wwlLabel.setFont(QFont("Arial", 12))
        self.wwlLabel.setAlignment(Qt.AlignCenter)
        self.wwlLabel.setText(f"WW: {self.windowWidth}, WL: {self.windowLevel}")
        self.wwlLabel.setStyleSheet("QLabel { margin: 5px; }")  
        self.colormap_hBox.addWidget(self.wwlLabel)

    def update_bounding_boxes(self, rect):
        if self.sender() == self.imgLabel_1.bounding_box:  # Axial plane
            self.imgLabel_2.bounding_box.rect.setRight(rect.top())  #moving y axis in axial moves x axis in sagittal( which is left and right)
            self.imgLabel_2.bounding_box.rect.setLeft(rect.bottom())  
            self.imgLabel_3.bounding_box.rect.setLeft(rect.left())  #moving x axis in axial moves x axis in coronal
            self.imgLabel_3.bounding_box.rect.setRight(rect.right())  

        elif self.sender() == self.imgLabel_2.bounding_box:  # Sagittal plane
            self.imgLabel_1.bounding_box.rect.setBottom(rect.left())  
            self.imgLabel_1.bounding_box.rect.setTop(rect.right())  
            self.imgLabel_3.bounding_box.rect.setTop(rect.top())  
            self.imgLabel_3.bounding_box.rect.setBottom(rect.bottom()) 
        elif self.sender() == self.imgLabel_3.bounding_box:  # Coronal plane
            self.imgLabel_1.bounding_box.rect.setLeft(rect.left())  
            self.imgLabel_1.bounding_box.rect.setRight(rect.right())  
            self.imgLabel_2.bounding_box.rect.setTop(rect.top())  
            self.imgLabel_2.bounding_box.rect.setBottom(rect.bottom())  

        self.imgLabel_1.bounding_box.updateHandlesPositions()
        self.imgLabel_2.bounding_box.updateHandlesPositions()
        self.imgLabel_3.bounding_box.updateHandlesPositions()

        self.imgLabel_1.update()
        self.imgLabel_2.update()
        self.imgLabel_3.update()

    def map_rect_to_plane(self, rect, source_plane, target_plane):
        # Map the rectangle coordinates from the source plane to the target plane
        if source_plane == 'axial':
            if target_plane == 'sagittal':
                return QRectF(rect.top(), 0, rect.height(), 511)
            elif target_plane == 'coronal':
                return QRectF(rect.left(), 0, rect.width(), 511)
        elif source_plane == 'sagittal':
            if target_plane == 'axial':
                return QRectF(0, rect.left(), 511, rect.height())
            elif target_plane == 'coronal':
                return QRectF(0, rect.top(), 511, rect.height())
        elif source_plane == 'coronal':
            if target_plane == 'axial':
                return QRectF(rect.left(), 0, rect.width(), 511)
            elif target_plane == 'sagittal':
                return QRectF(0, rect.top(), 511, rect.height())
            
    @staticmethod
    def adjust_image_based_on_ww_wl(img, ww, wl):
        lower_bound = wl - ww / 2
        upper_bound = wl + ww / 2
        img_adjusted = np.clip((img - lower_bound) / (upper_bound - lower_bound) * 255, 0, 255)
        return img_adjusted.astype(np.uint8)

    def toggle_slicer_functionality(self):

        self.boundingBox.setChecked(False)
        self.windowing.setChecked(True)
        self.toggleSlicerEnabled = True
        self.toggleBoundingBoxEnabled = False

        if self.toggleSlicerEnabled:
            self.imgLabel_1.setMouseTracking(True)
            self.imgLabel_2.setMouseTracking(True)
            self.imgLabel_3.setMouseTracking(True)
            self.imgLabel_1.type = 'axial'
            self.imgLabel_2.type = 'sagittal'
            self.imgLabel_3.type = 'coronal'
            self.boundingBox.setStyleSheet('''
                                           QPushButton{
                                            background-color: none;
                                            border-style: solid;
                                            border-color: #A7B4BF;
                                            border-width: 1px;
                                            color: #A7B4BF;
                                            text-align: center;
                                            padding: 8px;
                                            font: bold 12px;
                                            min-width: 10em;
                                            border-radius: 15px;
                                        }
                                           ''')
            self.windowing.setStyleSheet('''
                                         QPushButton {
                                        background-color: #0E6AC2;
                                        border-style: solid;
                                        border-color: #A7B4BF;
                                        border-width: 1px;
                                        color: white;
                                        text-align: center;
                                        padding: 8px;
                                        font: bold 12px;
                                        min-width: 10em;
                                        border-radius: 15px;
                                        }
                                        ''')
        else:
            self.imgLabel_1.setMouseTracking(False)
            self.imgLabel_2.setMouseTracking(False)
            self.imgLabel_3.setMouseTracking(False)
            self.imgLabel_1.type = 'axial'
            self.imgLabel_2.type = 'sagittal'
            self.imgLabel_3.type = 'coronal'
        self.updateimg()

        self.generateMask.setEnabled(False)
        self.generateMask.setStyleSheet('''
                                        QPushButton{
                                            background-color: #D9D9D9;
                                            border: none;
                                            color: white;
                                            text-align: center;
                                            padding: 10px;
                                            font: bold 12px;
                                            min-width: 10em;
                                            border-radius: 16px;
                                        }
                                        ''')

    def toggle_bounding_box_functionality(self):
        self.boundingBox.setChecked(True)
        self.windowing.setChecked(False)
        self.toggleSlicerEnabled = False
        self.toggleBoundingBoxEnabled = True

        if self.toggleBoundingBoxEnabled:
            self.imgLabel_1.type = 'axial'
            self.imgLabel_2.type = 'sagittal'
            self.imgLabel_3.type = 'coronal'
            self.windowing.setStyleSheet('''
                                           QPushButton{
                                            background-color: none;
                                            border-style: solid;
                                            border-color: #A7B4BF;
                                            border-width: 1px;
                                            color: #A7B4BF;
                                            text-align: center;
                                            padding: 8px;
                                            font: bold 12px;
                                            min-width: 10em;
                                            border-radius: 15px;
                                        }
                                           ''')
            self.boundingBox.setStyleSheet('''
                                         QPushButton {
                                        background-color: #0E6AC2;
                                        border-style: solid;
                                        border-color: #A7B4BF;
                                        border-width: 1px;
                                        color: white;
                                        text-align: center;
                                        padding: 8px;
                                        font: bold 12px;
                                        min-width: 10em;
                                        border-radius: 15px;
                                        }
                                        ''')
        else:
            self.imgLabel_1.type = 'axial'
            self.imgLabel_2.type = 'sagittal'
            self.imgLabel_3.type = 'coronal'
        self.updateimg()

        # enable generateMask Button
        self.generateMask.setEnabled(True)
        self.generateMask.setStyleSheet('''
                                        QPushButton {
                                            background-color: #013769;
                                            border: none;
                                            color: white;
                                            text-align: center;
                                            padding: 10px;
                                            font: bold 12px;
                                            min-width: 10em;
                                            border-radius: 16px;
                                        }
                                        ''')

    def set_directory(self):
        os.chdir(self.directory)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.w = self.imgLabel_1.width()
        self.h = self.imgLabel_1.height()

        # # Print the sizes of imgLabel_1, imgLabel_2, and imgLabel_3
        # print("imgLabel_1 size: width =", self.imgLabel_1.width(), "height =", self.imgLabel_1.height())
        # print("imgLabel_2 size: width =", self.imgLabel_2.width(), "height =", self.imgLabel_2.height())
        # print("imgLabel_3 size: width =", self.imgLabel_3.width(), "height =", self.imgLabel_3.height())
                
        if self.processedvoxel is not None:
            self.updateimg()

    def open_3dview(self):
        self.volWindow.setWindowTitle('3D View')
        self.volWindow.vol_show()
        self.volWindow.show()
        print("called")

    def dicom_clicked(self):
        dname = QFileDialog.getExistingDirectory(self, 'choose dicom directory')
        self.imgLabel_1.dname = dname
        self.imgLabel_2.dname = dname
        self.imgLabel_3.dname = dname
        
        self.load_dicomfile(dname)

        self.windowing.setEnabled(True)
        self.boundingBox.setEnabled(True)


    def load_dicomfile(self, dname):
        self.dcmList.clear()
        patient = ldf.load_scan(dname)
        imgs = ldf.get_pixels_hu(patient)
        self.voxel = self.linear_convert(imgs)
        self.processedvoxel = self.voxel.copy().astype(np.uint8)
        self.origin_processedvoxel = self.voxel.copy().astype(np.uint8)

        # self.processdvoxel (N, H, W)
        # print("size", self.processedvoxel.shape) # ex. (277, 512, 512)
        ###########################################################################################
        ### Get image embedding ???
        # CF inference_3D.py
        embedding_dim = (1, 256, 64, 64) 
        print("getting embedding....")
        # self.embedding = np.zeros((self.origin_processedvoxel.shape[0],) + embedding_dim, dtype=np.float32)
        self.embedding = torch.zeros((self.origin_processedvoxel.shape[0],) + embedding_dim, dtype=torch.float32, device=device)
  
        for i in range(self.origin_processedvoxel.shape[0]):
            img_2d = self.origin_processedvoxel[i, :, :]
            img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)  # (H, W, 3)

            # MedSAM Lite preprocessing
            img_256 = resize_longest_side(img_3c, 256)
            newh, neww = img_256.shape[:2]
            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )
            img_256_padded = pad_image(img_256, 256)
            img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
        
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
            
            self.embedding[i,:, :, :] = image_embedding
        print("Done!")
        ###########################################################################################

        self.update_shape()

        self.imgLabel_1.setMouseTracking(True)
        self.imgLabel_2.setMouseTracking(True)
        self.imgLabel_3.setMouseTracking(True)

        self.imgLabel_1.image_loaded = True
        self.imgLabel_2.image_loaded = True
        self.imgLabel_3.image_loaded = True
 
        self.updateimg()
        self.set_directory()
        self.volWindow = C3dView()
        self.volWindow.imgs = imgs
        self.volWindow.patient = patient
        self.dcmInfo = ldf.load_dcm_info(dname, False)
        self.image_loaded = True
        self.updatelist()

    def update_shape(self):
        self.v1, self.v2, self.v3 = self.processedvoxel.shape
        self.sagittal_vSlider.setMaximum(self.v1-1)
        self.coronal_vSlider.setMaximum(self.v1-1)
        self.sagittal_hSlider.setMaximum(self.v2-1)
        self.axial_vSlider.setMaximum(self.v2-1)
        self.coronal_hSlider.setMaximum(self.v3-1)
        self.axial_hSlider.setMaximum(self.v3-1)
        self.sagittal_vSlider.setValue(self.sagittal_vSlider.maximum()//2)
        self.coronal_vSlider.setValue(self.coronal_vSlider.maximum()//2)
        self.sagittal_hSlider.setValue(self.sagittal_hSlider.maximum()//2)
        self.axial_vSlider.setValue(self.axial_vSlider.maximum()//2)
        self.coronal_hSlider.setValue(self.coronal_hSlider.maximum()//2)
        self.axial_hSlider.setValue(self.axial_hSlider.maximum()//2)

    def updatelist(self):
        for item in self.dcmInfo:
            # 單純字串的話，可以不需要QListWidgetItem包裝也沒關係
            self.dcmList.addItem(QListWidgetItem('%-20s\t:  %s' % (item[0], item[1])))

    def updateimg(self):

        a_loc = self.sagittal_vSlider.value()
        c_loc = self.axial_vSlider.value()
        s_loc = self.axial_hSlider.value()

        axial = (self.processedvoxel[a_loc, :, :]).astype(np.uint8).copy()
        sagittal = (self.processedvoxel[:, :, s_loc]).astype(np.uint8).copy()
        
        coronal = (self.processedvoxel[:, c_loc, :]).astype(np.uint8).copy()

        self.imgLabel_1.slice_loc = [s_loc, c_loc, a_loc]
        self.imgLabel_2.slice_loc = [s_loc, c_loc, a_loc]
        self.imgLabel_3.slice_loc = [s_loc, c_loc, a_loc]

        if self.cross_recalc:
            self.imgLabel_1.crosscenter = [self.w*s_loc//self.v3, self.h*c_loc//self.v2]
            self.imgLabel_2.crosscenter = [self.w*c_loc//self.v2, self.h*a_loc//self.v1]
            self.imgLabel_3.crosscenter = [self.w*s_loc//self.v3, self.h*a_loc//self.v1]

        if self.colormap is None:
            self.imgLabel_1.processedImage = axial
            self.imgLabel_2.processedImage = sagittal
            self.imgLabel_3.processedImage = coronal
        else:
            self.imgLabel_1.processedImage = cv2.applyColorMap(axial, self.colormap)
            self.imgLabel_2.processedImage = cv2.applyColorMap(sagittal, self.colormap)
            self.imgLabel_3.processedImage = cv2.applyColorMap(coronal, self.colormap)

        axial_adjusted = self.adjust_image_based_on_ww_wl(axial, self.windowWidth, self.windowLevel)
        sagittal_adjusted = self.adjust_image_based_on_ww_wl(sagittal, self.windowWidth, self.windowLevel)
        coronal_adjusted = self.adjust_image_based_on_ww_wl(coronal, self.windowWidth, self.windowLevel)

        # Update processedImage for each label
        self.imgLabel_1.processedImage = axial_adjusted
        self.imgLabel_2.processedImage = sagittal_adjusted
        self.imgLabel_3.processedImage = coronal_adjusted

        # Display the adjusted images
        self.imgLabel_1.display_image(1)
        self.imgLabel_2.display_image(1)
        self.imgLabel_3.display_image(1)
        
        # Update the WW and WL label
        self.wwlLabel.setText(f"WW: {self.windowWidth}, WL: {self.windowLevel}")

        # Update the display of the segmentation result on the axial, sagittal, and coronal planes
        if self.segmentation_result is not None:
            axial_seg = self.segmentation_result[a_loc, :, :].astype(np.uint8) * 255
            sagittal_seg = self.segmentation_result[:, :, s_loc].astype(np.uint8) * 255
            coronal_seg = self.segmentation_result[:, c_loc, :].astype(np.uint8) * 255

            # Create color masks for each plane
            axial_mask = np.zeros((axial_seg.shape[0], axial_seg.shape[1], 3), dtype=np.uint8)
            axial_mask[axial_seg > 0] = (0, 255, 0)  # Green color for segmentation

            sagittal_mask = np.zeros((sagittal_seg.shape[0], sagittal_seg.shape[1], 3), dtype=np.uint8)
            sagittal_mask[sagittal_seg > 0] = (0, 255, 0)  # Green color for segmentation

            coronal_mask = np.zeros((coronal_seg.shape[0], coronal_seg.shape[1], 3), dtype=np.uint8)
            coronal_mask[coronal_seg > 0] = (0, 255, 0)  # Green color for segmentation

            # Overlay the color masks on the original images
            axial_overlaid = cv2.addWeighted(cv2.cvtColor(axial_adjusted, cv2.COLOR_GRAY2BGR), 1, axial_mask, 0.3, 0)
            sagittal_overlaid = cv2.addWeighted(cv2.cvtColor(sagittal_adjusted, cv2.COLOR_GRAY2BGR), 1, sagittal_mask, 0.3, 0)
            coronal_overlaid = cv2.addWeighted(cv2.cvtColor(coronal_adjusted, cv2.COLOR_GRAY2BGR), 1, coronal_mask, 0.3, 0)

            # Update processedImage for each label with the overlaid images
            self.imgLabel_1.processedImage = axial_overlaid
            self.imgLabel_2.processedImage = sagittal_overlaid
            self.imgLabel_3.processedImage = coronal_overlaid
        else:
            # If segmentation result is not available, use the adjusted images
            self.imgLabel_1.processedImage = axial_adjusted
            self.imgLabel_2.processedImage = sagittal_adjusted
            self.imgLabel_3.processedImage = coronal_adjusted

        # Display the images
        self.imgLabel_1.display_image(1)
        self.imgLabel_2.display_image(1)
        self.imgLabel_3.display_image(1)


    def generateEvent(self):
        ###################################################################################
        # When user press generate button, start generating mask
        # Using axial bounding box, axial image embedding with medsam_inference
        # 
        # 
        ###################################################################################

        print("Generate")
        if self.imgLabel_1.bounding_box is not None: # and self.imgLabel_2.bounding_box is not None and self.imgLabel_3.bounding_box is not None:
            # Get the bounding box coordinates from each plane
            axial_box = self.imgLabel_1.bounding_box.rect
            sagittal_box = self.imgLabel_2.bounding_box.rect
            #coronal_box = self.imgLabel_3.bounding_box.rect
            zmin = min(sagittal_box.top(), sagittal_box.bottom())
            zmax = max(sagittal_box.top(), sagittal_box.bottom())

            # Convert the bounding box coordinates to the appropriate format
            xmin = min(axial_box.left(), axial_box.right())
            xmax = max(axial_box.left(), axial_box.right())
            ymin = min(axial_box.top(), axial_box.bottom())
            ymax = max(axial_box.top(), axial_box.bottom())
         
            box_np = np.array([[xmin, ymin, xmax, ymax]])
            N, H, W = self.origin_processedvoxel.shape[:]
            box_256 = box_np / np.array([W, H, W, H]) * 256
         
            zstart = int(zmin / 512 * N)
            zend = int(zmax / 512 * N)

            for i in range(N):
                
                img_2d = self.origin_processedvoxel[i, :, :]
                if i >= zstart and i <= zend:
                    sam_mask = medsam_inference(medsam_lite_model, self.embedding[i], box_256, H, W)

                    mask_c = np.zeros((H,W), dtype="uint8") # (512, 512)
                
                    mask_c[sam_mask != 0] = 255
                    # self.origin imabe + self.mask => masked_image
                    masked_image = cv2.add(img_2d, mask_c)
            

                    # Update the processedvoxel with the masked image
                    self.processedvoxel[i, :, :] = masked_image
                
                else:
                    self.processedvoxel[i, :, :] = img_2d
           
            # Update the segmentation result
            print("segmentation end")
            
            # Update the display
            self.updateimg()
            print("Update end")

    @staticmethod
    def linear_convert(img):
        convert_scale = 255.0 / (np.max(img) - np.min(img))
        converted_img = convert_scale * img - (convert_scale * np.min(img))
        return converted_img

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CthreeD()
    ex.show()
    sys.exit(app.exec_())
