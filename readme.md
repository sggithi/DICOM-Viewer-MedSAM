<p align="center">
  <a href="https://github.com/wenyalintw/Dicom_Viewer">
    <img src="resources/brain.png" alt="Dicom Viewer" width="96" height="96">
  </a>
  <h2 align="center">DicomViewer GUI (Dicom Viewer)</h2>
  <p align="center"> 3D Dicom Viewer</p>
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
~~~
python run mainwindow.py
~~~
Load Checkpoint in work_dir/LiteMedSAM

base checkpoint(from bowang-lab/MedSAM) https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN checkpoint 저장
Fine Tuned Checkpoint (https://drive.google.com/drive/folders/1h0RgLM06RTMNbeLLVMD5opFw4kuaFNnG?usp=drive_link)


Strictly follow the package version in requirements.txt is not necessary.

## Acknowledgments
- Dicom Viewer reconstruction modified from [wenyalintw's Repo]([https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/](https://github.com/wenyalintw/Dicom-Viewer))
- MedSAM from Bowang-lab(https://github.com/bowang-lab/MedSAM)
###### INFINITT HealthCare (2024)
###### SNU CREATIVE INTEGRATED DESIGN 2  (2024), Hana Oh, Jimin Seo, Jahyun Yun
