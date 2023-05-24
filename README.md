# 3D-vis-reasoning

A stero vision-based 3D reconstruction and Recognition System,
originally developed under the scope of ECE188, Spring 2023, UCLA.



## Data Processing

Denoise the input image data, with the following tools:
1. classic denoise filters, [Done]
2. deblurring with blur kernal estimation from chen's code, [InProgress]
3. deeplearning method from Restromer. [Planned]



## 3D scene Reconstruction with 2D images
[Planning]
1. Capture Stereo Images: First, you need two images of the same scene taken from two different perspectives. This is typically done with a stereo camera setup where two cameras are placed a certain distance apart (known as the baseline).

2. Rectify Images: The images are then rectified to align corresponding points on the same horizontal line. This simplifies the process of finding corresponding points in the two images.

3. Compute Disparity Map: The disparity map is computed by finding the difference in the x-coordinates of corresponding points in the two images. This can be done using various algorithms such as block matching, semi-global block matching, or graph cuts.

4. Calculate Depth: Once the disparity map is computed, the depth can be calculated using the formula:

Z = \frac{fB}{d}
​
where:

Z is the depth (distance from the camera to the point)
f is the focal length of the camera
B is the baseline (distance between the two cameras)
d is the disparity (difference in x-coordinates of a point in the two images)
Generate Depth Map: The depth information can then be used to generate a depth map, which is a 2D image that uses grayscale to represent depth (darker shades represent closer objects, and lighter shades represent farther objects).



## 3D point cloud Recognition
[...]