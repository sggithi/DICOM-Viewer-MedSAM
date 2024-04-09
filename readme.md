<p align="center">
  <a href="https://github.com/wenyalintw/Dicom_Viewer">
    <img src="resources/brain.png" alt="Dicom Viewer" width="96" height="96">
  </a>
  <h2 align="center">DicomViewer GUI (Dicom Viewer)</h2>
  <p align="center"> 2D/3D Dicom 영상 활용 가능</p>
  <br>
</p>

## 실행 화면
Main Window를 열면，좌측 상단의 메뉴에서 2D processing와 3D processing 선택 항목이 있습니다. 그 중 3D processing 메뉴에는 3D volume reconstruction 기능이 있습니다.

### Main Window
<a href="https://github.com/wenyalintw/Dicom_Viewer">
    <img src="resources/mainwindow.png" alt="mainwindow" width="960" height="480">
</a>

### 2D processing
포함하는 기능은 다음과 같습니다.
- Load Image
- Save Image
- Convert to gray scale
- Restore
- Thresholding
- Region Growing
- Morthology (Dilation, Erosion, Opening, Closing)
- Edge Detection (Laplacian, Sobel, Perwitt, Frei & Chen)
- Drawing
<br>
<a href="https://github.com/wenyalintw/Dicom_Viewer">
    <img src="resources/2D_Processing.jpg" alt="2D_Processing" width="960" height="480">
</a>


### 3D processing
포함하는 기능은 다음과 같습니다.
- Load DICOM stack
- Save slice (axial, sagittal, coronal)
- Colormap transform
- Slider scrolling
- Mouse hovering/clicking
- Show DICOM info
- Show slice index coordinate
- 3D volume reconstruction
<br>
<a href="https://github.com/wenyalintw/Dicom_Viewer">
    <img src="resources/3D_Processing.jpg" alt="3D_Processing" width="960" height="480">
</a>

### 3D volume reconstruction
<br>
<a href="https://github.com/wenyalintw/Dicom_Viewer">
    <img src="resources/3D_Volume.jpg" alt="3D_Volume" width="480" height="480">
</a>

## How to use it?
Project root will be **/src**, just clone it and run mainwindow.py.



Strictly follow the package version in requirements.txt is not necessary.

## Acknowledgments
- [brain.png](https://github.com/wenyalintw/Dicom-Viewer/blob/master/resources/brain.png) licensed under "CC BY 3.0" downloaded from [ICONFINDER](https://www.iconfinder.com/icons/1609653/brain_organs_icon) 
- 3D volumn reconstruction modified from [Howard Chen's Post](https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/)

###### MIT License (2019), Wen-Ya Lin
