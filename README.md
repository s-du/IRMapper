# IR Mapper

## Introduction

This is a Pyside6 application for creating thermal pointclouds from an RGB/IR image folder, using photogrammetry.
Note: At the moment, the application is compatible with the following drones models:
- DJI Mavic 2 Enterprise Advanced
- DJI Mavic 3T

The application uses **Agisoft Metashape Python API** but we are working on Open Drone Map integration!

<a href="https://ibb.co/MfKBYKL"><img src="https://i.ibb.co/ScHfhHp/Thermal-Mesh.png" alt="Thermal-Mesh" border="0"></a>

*GUI for thermal photogrammetry*

## Principle
High-resolution reconstructions with integrated infrared data can be generated for the exterior of the buildings, but also for the interior. In addition to highlighting potential thermal weaknesses of the envelope, the processing of such point clouds also allows new types of analysis.

![Picture1.png](https://i.postimg.cc/ryBngFn4/Picture1.png)

Here, from a DJI image folder (typically ..._T.JPEG alternating with ..._W.JPEG), the application will guide the user to reconstruct a infrared point cloud.

### Step 1: Processing of images
First, the collected IR images have to be modified before entering the photogrammetry process. This phase was essential because of the highly processed nature of the R-JPG images coming out of the DJI M2EA/ DJI M3T drone. The temperature scale is the most critical parameter here. In order to guarantee an optimal integration of the thermal information into the 3D process, it is imperative to have a fixed temperature scale on the whole photoset (which is not the case by default). **This app allows batch processing of infrared images**, making photosets more easily compatible with photogrammetric processes.

### Step 2: 3D reconstruction
From infrared images with a consistant temperature range, it is possibile to reconstruct accurate 3D models. First, this implies that **pairs of images** are captured: Colour + Infrared images at each shot. This is the default behaviour of DJI thermal drones. Secondly, the photogrammetric process requires to have a significant overlap between individual shots. When regular RGB pictures are used, 75% overlap can be judged satisfactory. With IR/RGB pairs, it is advised to increase the overlap, because the field of view is smaller on IR pictures. 

<a href="https://imgbb.com/"><img src="https://i.ibb.co/T8HKHVP/field.png" alt="field" border="0"></a>

*Difference in terms of field of view between thermal and colour pictures*

## Features
The app offers the following key features:

1. Image Loading:
    - Users can easily load thermal images by importing a folder with RGB/IR pairs, directly exported from the drone SD card.
  
2. Image Visualization:
    - The app provides tools to visualize the loaded thermal images, including zooming, panning, and scaling options.

3. Image Enhancement:
    - Users can adjust the temperature scale and colour palettes
    - Users can apply various image enhancement techniques to improve the quality and clarity of thermal images.
  
4. Image batch processing:
    - The app enables users to perform batch processing of thermal images. It allows to obtain a dataset compatible with 3D photogrammetry.

5. 3D reconstruction
    - Users can choose a thermal image set and create a 3D reconstruction of the captured zone. This phase relies on Agisoft Metashape API (https://www.agisoft.com/pdf/metashape_python_api_2_0_2.pdf)

6. 3D Visualization
    - Users can visualize the 3D reconstruction in a simple 3D viewer.
    - Some visual options: colour/infrared switch, type of render, ...

Upcoming key features:

**Image Analysis**:
    - Users can extract temperature data and generating temperature distribution maps from loaded images
    - The app allows users to define and select specific regions of interest within the thermal images for in-depth analysis.

**3D reconstruction**
    - Support added for WebODM 


## Installation instructions
A simple Windows setup file is available in the 'Releases' section!

Alternatively:
1. Clone the repository:
```
git clone https://github.com/s-du/IRMapper
```

2. Navigate to the app directory:
```
cd app_directory
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. Run the app:
```
python main.py
```

## User manual
(coming soon)

## Contributing

Contributions to the IRMapper App are welcome! If you find any bugs, have suggestions for new features, or would like to contribute enhancements, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request describing your changes.

## Acknowledgements
This project was made possible thanks to subsidies from the Brussels Capital Region, via Innoviris.
Feel free to use or modify the code, in which case you can cite Buildwise and the Pointify project!

## TO DO

- [x] ~~Implement thread system for progress bar~~
- [ ] Add a temperature legend in image processing
- [ ] Standardized image processing methods (opencv / pillow / other)
- [ ] Standardize naming convention (eg. ir or th)
- [ ] Extensively document the code
- [ ] Add Open Drone Map Support
- [ ] Add Open CV Weka (--> Developed on the side)

Main icon:
<a href="https://www.flaticon.com/free-icons/miscellaneous" title="miscellaneous icons">Miscellaneous icons created by Vectors Market - Flaticon</a>

![wtc-00001-l-horizontaal-pos-rgb.png](https://i.postimg.cc/zDy3VjNJ/wtc-00001-l-horizontaal-pos-rgb.png)
