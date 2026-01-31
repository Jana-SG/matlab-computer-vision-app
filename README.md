# Advanced Computer Vision Toolkit (MATLAB)

A comprehensive GUI-based application built with MATLAB App Designer that implements a broad set of classical image processing and computer vision algorithms. The system provides a unified graphical interface for experimenting with filtering, feature extraction, robust model fitting, and stereo vision techniques.


## ✨ Key Features

### 📁 **Core Modules:**
- **Basic Operations**: Grayscale conversion, binarization, color space transformations (RGB, HSI, Lab, YCbCr)
- **Spatial Domain Filtering**: Box, weighted average, median, Laplacian, Sobel, Prewitt filters
- **Frequency Domain Filtering**: Butterworth & Gaussian low/high pass filters
- **Pyramids & Template Matching**: Gaussian/Laplacian pyramids, SSD, normalized cross-correlation
- **Edge & Corner Detection**: Canny edge detection, Harris corner detection
- **Feature Descriptors**: HOG, LM filter banks, DoG, LoG
- **Hough Transforms**: Line and circle detection with configurable parameters
- **RANSAC**: Robust line/circle fitting and feature matching
- **Stereo Vision**: Epipolar geometry, fundamental matrix estimation

### 🔧 Technical Highlights
- Modular architecture with 11 specialized processing modules
- Interactive parameter tuning with real-time visualization
- Custom implementations of RANSAC, LM filter banks, and template matching
- Dual-view visualization (original vs. processed image)
- Integration of classical CV pipelines in a single application
  
## 📋 Prerequisites
- MATLAB R2020b or later
- Image Processing Toolbox
- Computer Vision Toolbox
- Statistics and Machine Learning Toolbox (for geometric model estimation functions)

## How to use 
1. Clone repository: `git clone [repo-url]`
2. Open `ComputerVisionToolBox.mlapp` in MATLAB
3. Click **Run** in App Designer
4. Click "Load Image" to select any image
5. Navigate to any tab (e.g., "Edge and Corner")
6. Adjust parameters and see real-time updates
7. Original vs. modified images shown side-by-side

