# IR Mapper

This is a Pyside6 application for creating thermal pointclouds from a RGB/IR image folder, using photogrammetry.
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

## Installation instructions
For advanced users, download the code and install Python requirements.
A simple Windows setup file is also available in the 'Releases' section!

## User manual
(coming soon)

## Acknowledgements
This project was made possible thanks to subsidies from the Brussels Capital Region, via Innoviris.
Feel free to use or modify the code, in which case you can cite Buildwise and the Pointify project!

## TO DO

- [ ] Implement thread system for progress bar (in progress)
- [ ] Add a temperature legend in image processing
- [ ] Standardized image processing methods (opencv / pillow / other)
- [ ] Standardize naming convention (eg. ir or th)
- [ ] Document code
- [ ] Add Open Drone Map Support
- [ ] Switch to Potree for point cloud visualization
- [ ] Add Open CV Weka (--> Developed on the side)

Main icon:
<a href="https://www.flaticon.com/free-icons/miscellaneous" title="miscellaneous icons">Miscellaneous icons created by Vectors Market - Flaticon</a>

![wtc-00001-l-horizontaal-pos-rgb.png](https://i.postimg.cc/zDy3VjNJ/wtc-00001-l-horizontaal-pos-rgb.png)
