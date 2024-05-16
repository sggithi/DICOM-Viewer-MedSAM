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

from .inference_3D import medsam_lite_model, medsam_inference, resize_longest_side, pad_image
# with torch.no_grad():
#     img_256_tensor = torch.tensor(self.processedvoxel).float().permute(0, 3, 1, 2).to(device)
#     self.embedding = medsam_lite_model.image_encoder(img_256_tensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
Overall structure of the CthreeD class:
- The class inherits from QDialog
- The class has the following methods:

    ############################################################################################################
    Initializers:
    - __init__(): Initializes the class
    - setup_paths(): Sets the current working directory to the 'threeD' directory
    - load_ui(): Loads the UI from the 'threeD_module.ui' file
    - initialize_variables(): Initializes the class variables
    - setup_ui_components(): Sets up the UI components of the class
    - set_grid_spacing(): Sets the spacing for a grid layout
    - set_vbox_spacing(): Sets the spacing for a vertical box layout
    - set_hbox_spacing(): Sets the spacing for a horizontal box layout
    - setup_buttons(): Sets up the functionality for the buttons in the UI
    - get_button_style(): Returns the style for a button based on the color
    - initialize_labels(): Initializes the labels for the images
    - initialize_sliders(): Initializes the sliders for the images
    ############################################################################################################

    ############################################################################################################
    Image Processing:

    Bounding Box Handling:
    - update_bounding_boxes(): Updates the bounding boxes for the images
    - map_rect_to_plane(): Maps the rectangle coordinates from the source plane to the target plane

    HU windowing:
    - adjust_image_based_on_ww_wl(): Adjusts the image based on the window width and window level

    UI Toggle Functions:
    - toggle_slicer_functionality(): Toggles the slicer functionality
    - toggle_bounding_box_functionality(): Toggles the bounding box functionality

    DICOM Handling:
    - dicom_clicked(): Opens the DICOM files from the selected directory
    - load_dicomfile(): Loads the DICOM files from the selected directory
    - set_directory(): Sets the current working directory to the directory variable
    - resizeEvent(): Resizes the event based on the window size
    - open_3dview(): Opens the 3D view window

    Image Embedding and Label Initialization:
    - get_image_embeddings(): Gets the image embeddings using the MedSAM Lite model
    - enable_image_labels(): Enables the image labels
    - update_shape(): Updates the shape of the voxel
    - set_slider_values_to_middle(): Sets the slider values to the middle
    - updatelist(): Updates the DICOM information list

    Image Update and Display:
    - updateimg(): Updates the images based on the current slider values
    - set_slice_locations(): Sets the slice locations for the images
    - update_cross_centers(): Updates the cross centers for the images
    - update_processed_images(): Updates the processed images for the images
    - display_images(): Displays the images
    - update_segmentation_result(): Updates the segmentation result for the images
    - overlay_segmentation(): Overlays the segmentation on the image
    ############################################################################################################

    ############################################################################################################
    Segmentation:
    - generateEvent(): Generates the segmentation masks using the current bounding box and image embeddings
    - generate_masks_for_slices(): Generates the masks for the slices
    - linear_convert(): Converts the image to a linear scale
    ############################################################################################################
"""

class CthreeD(QDialog):

    def __init__(self):
        super().__init__()
        self.setup_paths()
        self.load_ui()
        self.initialize_variables()
        self.setup_ui_components()

    def setup_paths(self):
        path = os.getcwd()
        os.chdir(path + '/threeD')
        self.directory = os.getcwd()
    
    def load_ui(self):
        loadUi('threeD_module.ui', self)
        self.setWindowTitle('3D Processing')

    def initialize_variables(self):
        self.image = None
        self.voxel = None
        self.processedvoxel = None
        self.v1, self.v2, self.v3 = None, None, None
        self.volWindow = None
        self.colormap = None
        self.cross_recalc = True
        self.box_coordinates = []
        self.segmentation_result = None
        self.origin_processedvoxel = None
        self.windowWidth = 250  # Default window width
        self.windowLevel = 50   # Default window level
        self.toggleSlicerEnabled = False
        self.toggleBoundingBoxEnabled = True
        self.w = self.imgLabel_1.width()
        self.h = self.imgLabel_1.height()

    def setup_ui_components(self):
        self.imgLabel_1.type = 'axial'
        self.imgLabel_2.type = 'sagittal'
        self.imgLabel_3.type = 'coronal'
        self.imgLabel_1.setFixedSize(512, 512)
        self.imgLabel_2.setFixedSize(512, 512)
        self.imgLabel_3.setFixedSize(512, 512)
        self.setup_spacing()
        self.setup_buttons()
        self.initialize_labels()
        self.initialize_sliders()
        self.dcmInfo = None

        # This spacer will push everything to the left of it to the left, and everything to the right of it to the right
        self.colormap_hBox.addStretch(1)
        self.wwlLabel = QLabel(self)
        self.wwlLabel.setFont(QFont("Arial", 12))
        self.wwlLabel.setAlignment(Qt.AlignCenter)
        self.wwlLabel.setText(f"WW: {self.windowWidth}, WL: {self.windowLevel}")
        self.wwlLabel.setStyleSheet("QLabel { margin: 5px; }")  
        self.colormap_hBox.addWidget(self.wwlLabel)

    def setup_spacing(self):
        self.set_grid_spacing(self.axialGrid)
        self.set_grid_spacing(self.saggitalGrid)
        self.set_grid_spacing(self.coronalGrid)
        self.set_vbox_spacing(self.axial_vBox)
        self.set_vbox_spacing(self.saggital_vBox)
        self.set_vbox_spacing(self.coronal_vBox)
        self.set_hbox_spacing(self.axial_hBox)
        self.set_hbox_spacing(self.saggital_hBox)
        self.set_hbox_spacing(self.coronal_hBox)
        self.colormap_hBox.insertStretch(2)
        self.colormap_hBox.insertSpacerItem(0, QSpacerItem(30, 0, QSizePolicy.Fixed,  QSizePolicy.Fixed))

    def set_grid_spacing(self, grid):
        grid.setSpacing(1)

    def set_vbox_spacing(self, vbox):
        spacer = QSpacerItem(10, 10, QSizePolicy.Fixed, QSizePolicy.Fixed)
        vbox.setSpacing(0)
        vbox.insertSpacerItem(0, spacer)
        vbox.insertSpacerItem(2, spacer)

    def set_hbox_spacing(self, hbox):
        spacer = QSpacerItem(10, 10, QSizePolicy.Fixed, QSizePolicy.Fixed)
        hbox.setSpacing(0)
        hbox.insertSpacerItem(0, spacer)
        hbox.insertSpacerItem(2, spacer)

    def setup_buttons(self):
        self.dicomButton.clicked.connect(self.dicom_clicked)
        self.volButton.clicked.connect(self.open_3dview)
        self.boundingBox.clicked.connect(self.toggle_bounding_box_functionality)
        self.boundingBox.setChecked(True)
        self.boundingBox.setStyleSheet(self.get_button_style("#B6C2CE"))
        self.windowing.clicked.connect(self.toggle_slicer_functionality)
        self.windowing.setChecked(False)
        self.windowing.setEnabled(False)
        self.boundingBox.setEnabled(False)
        self.generateMask.setEnabled(False)
        self.generateMask.clicked.connect(self.generateEvent)

    def get_button_style(self, color):
        return f'''
        QPushButton {{
            background-color: {color};
            border-style: solid;
            border-color: #A7B4BF;
            border-width: 1px;
            color: white;
            text-align: center;
            padding: 8px;
            font: bold 12px;
            min-width: 10em;
            border-radius: 15px;
        }}'''

    def initialize_labels(self):
        self.imgLabel_1.updateNeeded.connect(self.updateimg)
        self.imgLabel_2.updateNeeded.connect(self.updateimg)
        self.imgLabel_3.updateNeeded.connect(self.updateimg)
        self.imgLabel_1.bounding_box_resized.connect(self.update_bounding_boxes)
        self.imgLabel_2.bounding_box_resized.connect(self.update_bounding_boxes)
        self.imgLabel_3.bounding_box_resized.connect(self.update_bounding_boxes)

    def initialize_sliders(self):
        self.axial_hSlider.valueChanged.connect(self.updateimg)
        self.axial_vSlider.valueChanged.connect(self.updateimg)
        self.sagittal_hSlider.valueChanged.connect(self.updateimg)
        self.sagittal_vSlider.valueChanged.connect(self.updateimg)
        self.coronal_hSlider.valueChanged.connect(self.updateimg)
        self.coronal_vSlider.valueChanged.connect(self.updateimg)

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
        # Toggle slicer functionality
        self.boundingBox.setChecked(False)
        self.windowing.setChecked(True)
        self.toggleSlicerEnabled = True
        self.toggleBoundingBoxEnabled = False
        self.update_button_styles()
        self.enable_mouse_tracking(True)
        self.generateMask.setEnabled(False)
        self.generateMask.setStyleSheet(self.get_button_style("#D9D9D9"))

    def toggle_bounding_box_functionality(self):
        # Toggle bounding box functionality
        self.boundingBox.setChecked(True)
        self.windowing.setChecked(False)
        self.toggleSlicerEnabled = False
        self.toggleBoundingBoxEnabled = True
        self.update_button_styles()
        self.enable_mouse_tracking(False)
        self.generateMask.setEnabled(True)
        self.generateMask.setStyleSheet(self.get_button_style("#013769"))


    def set_directory(self):
        os.chdir(self.directory)


    def update_button_styles(self):
        # Update button styles based on toggles
        self.boundingBox.setStyleSheet(self.get_button_style("#B6C2CE" if self.toggleBoundingBoxEnabled else "none"))
        self.windowing.setStyleSheet(self.get_button_style("#B6C2CE" if self.toggleSlicerEnabled else "none"))

    def enable_mouse_tracking(self, enable):
        # Enable or disable mouse tracking for image labels
        self.imgLabel_1.setMouseTracking(enable)
        self.imgLabel_2.setMouseTracking(enable)
        self.imgLabel_3.setMouseTracking(enable)

    def open_3dview(self):
        # Open the 3D view window
        self.volWindow = C3dView()
        self.volWindow.setWindowTitle('3D View')
        self.volWindow.vol_show()
        self.volWindow.show()

    def dicom_clicked(self):
        dname = QFileDialog.getExistingDirectory(self, 'choose dicom directory')
        self.imgLabel_1.dname = dname
        self.imgLabel_2.dname = dname
        self.imgLabel_3.dname = dname
        self.load_dicomfile(dname)
        self.windowing.setEnabled(True)
        self.boundingBox.setEnabled(True)


    def load_dicomfile(self, dname):
        # Load DICOM files from the selected directory
        self.dcmList.clear()
        patient = ldf.load_scan(dname)
        imgs = ldf.get_pixels_hu(patient)
        self.voxel = self.linear_convert(imgs)
        self.processedvoxel = self.voxel.copy().astype(np.uint8)
        self.origin_processedvoxel = self.voxel.copy().astype(np.uint8)
        self.embedding = self.get_image_embeddings()
        self.update_shape()
        self.enable_image_labels()
        self.updateimg()
        self.set_directory()
        self.volWindow = C3dView()
        self.volWindow.imgs = imgs
        self.volWindow.patient = patient
        self.dcmInfo = ldf.load_dcm_info(dname, False)
        self.image_loaded = True
        self.updatelist()

    def get_image_embeddings(self):
        # Get image embeddings using MedSAM Lite model
        embedding_dim = (1, 256, 64, 64)
        embeddings = torch.zeros((self.origin_processedvoxel.shape[0],) + embedding_dim, dtype=torch.float32, device=device)
        for i in range(self.origin_processedvoxel.shape[0]):
            img_2d = self.origin_processedvoxel[i, :, :]
            img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)  # (H, W, 3)
            img_256 = resize_longest_side(img_3c, 256)
            img_256 = (img_256 - img_256.min()) / np.clip(img_256.max() - img_256.min(), a_min=1e-8, a_max=None)
            img_256_padded = pad_image(img_256, 256)
            img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
            embeddings[i] = image_embedding
        return embeddings

    def enable_image_labels(self):
        # Enable mouse tracking and set flags for image labels
        self.imgLabel_1.setMouseTracking(True)
        self.imgLabel_2.setMouseTracking(True)
        self.imgLabel_3.setMouseTracking(True)
        self.imgLabel_1.image_loaded = True
        self.imgLabel_2.image_loaded = True
        self.imgLabel_3.image_loaded = True

    def update_shape(self):
        # Update slider maximum values based on voxel shape
        self.v1, self.v2, self.v3 = self.processedvoxel.shape
        self.sagittal_vSlider.setMaximum(self.v1 - 1)
        self.coronal_vSlider.setMaximum(self.v1 - 1)
        self.sagittal_hSlider.setMaximum(self.v2 - 1)
        self.axial_vSlider.setMaximum(self.v2 - 1)
        self.coronal_hSlider.setMaximum(self.v3 - 1)
        self.axial_hSlider.setMaximum(self.v3 - 1)
        self.set_slider_values_to_middle()

    def set_slider_values_to_middle(self):
        # Set sliders to their middle values
        self.sagittal_vSlider.setValue(self.sagittal_vSlider.maximum() // 2)
        self.coronal_vSlider.setValue(self.coronal_vSlider.maximum() // 2)
        self.sagittal_hSlider.setValue(self.sagittal_hSlider.maximum() // 2)
        self.axial_vSlider.setValue(self.axial_vSlider.maximum() // 2)
        self.coronal_hSlider.setValue(self.coronal_hSlider.maximum() // 2)
        self.axial_hSlider.setValue(self.axial_hSlider.maximum() // 2)

    def updatelist(self):
        # Update the DICOM information list
        for item in self.dcmInfo:
            # 單純字串的話，可以不需要QListWidgetItem包裝也沒關係
            self.dcmList.addItem(QListWidgetItem('%-20s\t:  %s' % (item[0], item[1])))

    def updateimg(self):
        # Update the displayed images based on the current slider values
        a_loc = self.sagittal_vSlider.value()
        c_loc = self.axial_vSlider.value()
        s_loc = self.axial_hSlider.value()
        axial = self.processedvoxel[a_loc, :, :].astype(np.uint8).copy()
        sagittal = self.processedvoxel[:, :, s_loc].astype(np.uint8).copy()
        coronal = self.processedvoxel[:, c_loc, :].astype(np.uint8).copy()
        self.set_slice_locations([s_loc, c_loc, a_loc])
        self.update_cross_centers(s_loc, c_loc, a_loc)
        axial_adjusted = self.adjust_image_based_on_ww_wl(axial, self.windowWidth, self.windowLevel)
        sagittal_adjusted = self.adjust_image_based_on_ww_wl(sagittal, self.windowWidth, self.windowLevel)
        coronal_adjusted = self.adjust_image_based_on_ww_wl(coronal, self.windowWidth, self.windowLevel)
        self.update_processed_images(axial_adjusted, sagittal_adjusted, coronal_adjusted)
        self.wwlLabel.setText(f"WW: {self.windowWidth}, WL: {self.windowLevel}")
        self.display_images()
        self.update_segmentation_result(a_loc, s_loc, c_loc, axial_adjusted, sagittal_adjusted, coronal_adjusted)

    def set_slice_locations(self, loc):
        self.imgLabel_1.slice_loc = loc
        self.imgLabel_2.slice_loc = loc
        self.imgLabel_3.slice_loc = loc

    def update_cross_centers(self, s_loc, c_loc, a_loc):
        if self.cross_recalc:
            self.imgLabel_1.crosscenter = [self.w * s_loc // self.v3, self.h * c_loc // self.v2]
            self.imgLabel_2.crosscenter = [self.w * c_loc // self.v2, self.h * a_loc // self.v1]
            self.imgLabel_3.crosscenter = [self.w * s_loc // self.v3, self.h * a_loc // self.v1]

    def update_processed_images(self, axial, sagittal, coronal):
        self.imgLabel_1.processedImage = axial
        self.imgLabel_2.processedImage = sagittal
        self.imgLabel_3.processedImage = coronal

    def display_images(self):
        self.imgLabel_1.display_image(1)
        self.imgLabel_2.display_image(1)
        self.imgLabel_3.display_image(1)

    def update_segmentation_result(self, a_loc, s_loc, c_loc, axial_adjusted, sagittal_adjusted, coronal_adjusted):
        if self.segmentation_result is not None:
            axial_seg = self.segmentation_result[a_loc, :, :].astype(np.uint8) * 255
            sagittal_seg = self.segmentation_result[:, :, s_loc].astype(np.uint8) * 255
            coronal_seg = self.segmentation_result[c_loc, :].astype(np.uint8) * 255
            axial_overlaid = self.overlay_segmentation(axial_adjusted, axial_seg)
            sagittal_overlaid = self.overlay_segmentation(sagittal_adjusted, sagittal_seg)
            coronal_overlaid = self.overlay_segmentation(coronal_adjusted, coronal_seg)
            self.update_processed_images(axial_overlaid, sagittal_overlaid, coronal_overlaid)
        else:
            self.update_processed_images(axial_adjusted, sagittal_adjusted, coronal_adjusted)
            self.display_images()

    def overlay_segmentation(self, img, seg):
        mask = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        mask[seg > 0] = (0, 255, 0)  # Green color for segmentation
        return cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 1, mask, 0.3, 0)

    def generateEvent(self):
        # Generate segmentation masks using the current bounding box and image embeddings
        print("Generate")
        if self.imgLabel_1.bounding_box is not None:
            axial_box = self.imgLabel_1.bounding_box.rect
            sagittal_box = self.imgLabel_2.bounding_box.rect
            zmin, zmax = min(sagittal_box.top(), sagittal_box.bottom()), max(sagittal_box.top(), sagittal_box.bottom())
            xmin, xmax = min(axial_box.left(), axial_box.right()), max(axial_box.left(), axial_box.right())
            ymin, ymax = min(axial_box.top(), axial_box.bottom()), max(axial_box.top(), axial_box.bottom())
            box_np = np.array([[xmin, ymin, xmax, ymax]])
            N, H, W = self.origin_processedvoxel.shape[:]
            box_256 = box_np / np.array([W, H, W, H]) * 256
            zstart, zend = int(zmin / 512 * N), int(zmax / 512 * N)
            self.generate_masks_for_slices(zstart, zend, box_256, H, W)
            print("segmentation end")
            self.updateimg()
            print("Update end")

    def generate_masks_for_slices(self, zstart, zend, box_256, H, W):
        for i in range(self.origin_processedvoxel.shape[0]):
            img_2d = self.origin_processedvoxel[i, :, :]
            if zstart <= i <= zend:
                sam_mask = medsam_inference(medsam_lite_model, self.embedding[i], box_256, H, W)
                mask_c = np.zeros((H, W), dtype="uint8")
                mask_c[sam_mask != 0] = 255
                masked_image = cv2.add(img_2d, mask_c)
                self.processedvoxel[i, :, :] = masked_image
            else:
                self.processedvoxel[i, :, :] = img_2d

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
