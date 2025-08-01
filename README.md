# Surgical Tooltip Localization via Concentric Nested Square Markers and Depth-RGB Multi-Coordinate Fusion
# Introduction
 This repository provides the implementation of a novel real-time surgical tooltip localization method based on concentric nested square markers and multi-coordinate frame fusion, as described in the paper "Surgical Tooltip Localization via Concentric Nested Square Markers and Depth-RGB Multi-Coordinate Fusion". The proposed method aims to improve the robustness and spatial accuracy of tooltip localization in surgical navigation and other high-precision applications.
# Features
1.Robust and accurate tooltip localization using concentric nested square markers.  
2.Multi-coordinate frame fusion for enhanced pose estimation.  
3.Optional depth data incorporation to further improve localization precision.  
4.Compatible with common camera systems.  
# Requirements
* Python 3.x  
* OpenCV  
* NumPy  
* Kinect v2 camera (or other RGB-D cameras with appropriate modifications)
# Citation
* Authors: Dexun Zhang†, Tianqiao Zhang†, Ahmed Elazab, Cong Li, Fucang Jia*, Huoling Luo*
* Title: Surgical Tooltip Localization via Concentric Nested Square Markers and Depth - RGB Multi Coordinate Fusion
* Journal: International Journal of Computer Assisted Radiology and Surgery
* DOI: https://doi.org/10.1007/s11548-025-03456-4
# Usage
Run the aruco_tooltip.py file  
👉Note that the code is designed to work with a Kinect v2 camera. If you are using a different camera, you may need to modify the image acquisition part of the code accordingly.  
![Example Image](picture/1-1.PNG)  
![Example Image](picture/2-2.PNG)  
![Example Image](picture/3-3.PNG)  




