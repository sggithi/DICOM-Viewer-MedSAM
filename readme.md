<p align="center">
  <a href="https://github.com/wenyalintw/Dicom_Viewer">
    <img src="resources/brain.png" alt="Dicom Viewer" width="96" height="96">
  </a>
  <h2 align="center">DicomViewer GUI (Dicom Viewer)</h2>
  <p align="center"> 2D/3D Dicom 영상 활용 가능</p>
  <br>
</p>


### Main Window
![스크린샷 2024-05-31 오후 9 49 48](https://github.com/sggithi/Dicom-Viewer-MedSAM/assets/52576276/a8e1f091-ad03-4e65-a644-35147a703a95)


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

![스크린샷 2024-05-31 오후 9 38 54](https://github.com/sggithi/Dicom-Viewer-MedSAM/assets/52576276/45a5db7e-f612-470d-b846-4bfe735893f4)


## How to use it?
Project root will be **/src**, just clone it and run mainwindow.py.
work_dir/MedSAM 폴더에 https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN checkpoint 저장


Strictly follow the package version in requirements.txt is not necessary.

## Acknowledgments
- [brain.png](https://github.com/wenyalintw/Dicom-Viewer/blob/master/resources/brain.png) licensed under "CC BY 3.0" downloaded from [ICONFINDER](https://www.iconfinder.com/icons/1609653/brain_organs_icon) 
- 3D volumn reconstruction modified from [Howard Chen's Post](https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/)

###### MIT License (2019), Wen-Ya Lin
###### INFINITT HealthCare (2024)
###### SNU 창의적통합설계, Hana Oh, Jimin Seo, Jahyun Yun (2024)
