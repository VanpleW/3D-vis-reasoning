# 3D-vis-reasoning

A stero vision-based 3D reconstruction and Recognition System,
originally developed under the scope of ECE188, Spring 2023, UCLA.

As in the repo structure below, all files named **demo.ipynb** are the ones for performance purpose, other files are suppliment functions.

- 3D-vis-reasoning
  - .vscode
    - settings.json
  - stage1
    - ref
    - src
    - stage1_data
    - .DS_Store
    - __init__.py
    - **demo.ipynb**
  - stage2
    - stage2_rectification
        - input_imgs
        - warped_imgs
        - __init__.py
        - **demo.ipynb**
        - training.log
        - utils.py
    - stage2_stereo
        - disparities
        - left_imgs
        - right_imgs
        - __init__.py
        - **demo.ipynb**
        - utils.py
    - .DS_Store
    - Final Project Part 2.pdf
  - stage3
    - **3Ddetection.ipynb**
  - .DS_Store
  - LICENSE
  - README.md


## Data Processing

Denoise the input image data, with the following tools:
1. classic denoise filters, [Done]
2. deblurring with blur kernal estimation from chen's code, [Done]
3. deeplearning method from Restromer. [Triedonline]


## 3D scene Reconstruction with 2D images

1. Capture Stereo Images: [Provided]
   
   First, you need two images of the same scene taken from two different perspectives. This is typically done with a stereo camera setup where two cameras are placed a certain distance apart (known as the baseline).

2. Rectify Images: [Done]
   The images are then rectified to align corresponding points on the same horizontal line. This simplifies the process of finding corresponding points in the two images.

3. Compute Disparity Map: [Done]
    The disparity map is computed by finding the difference in the x-coordinates of corresponding points in the two images. This can be done using various algorithms such as block matching, semi-global block matching, or graph cuts. Moreover, deeplearing based approach is also avaiable.

4. Calculate Depth: [Planned]
    Once the disparity map is computed, the depth can be calculated using the formula:

$$ Z = \frac{fB}{d} $$
​
where:

Z is the depth (distance from the camera to the point)
f is the focal length of the camera
B is the baseline (distance between the two cameras)
d is the disparity (difference in x-coordinates of a point in the two images)
Generate Depth Map: The depth information can then be used to generate a depth map, which is a 2D image that uses grayscale to represent depth (darker shades represent closer objects, and lighter shades represent farther objects).

5. Triangulate the 3D stereo image: [Planned]


## 3D point cloud Recognition

3D object detections Based on trainning pipeline provided in OpenMMlab/mmdetection3D.
Tested on colab pro.

- 3DSSD
- PV-RCNN
- PointRCNN
