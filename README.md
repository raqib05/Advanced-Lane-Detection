# 🚗 Advanced Lane Detection (Classical Computer Vision)

A **classical computer vision–based lane detection pipeline** for road images and videos, inspired by autonomous driving perception systems.  
This project performs **lane detection, polynomial fitting, bird’s-eye transformation, and inverse projection** using OpenCV and NumPy — without deep learning.

Depth detection will be implemented in the future
---

## 📌 Features

- Perspective (bird’s-eye) transformation using **homography**
- Edge + color-based lane extraction (Canny + HLS thresholding)
- Histogram-based **sliding window lane detection**
- Second-order polynomial fitting for curved lanes
- Lane region polygon generation
- Inverse perspective projection back onto the original frame
- Works on **both images and videos**

---

## 🧠 Pipeline Overview (processing.py)

1. **Perspective Transformation**
   - Homography transformation matrices are computed using cv2.getPerspectiveTransform
   - Convert front-facing road view into a top-down (bird’s-eye) view using homography

2. **Lane Feature Extraction**
   - Convert to grayscale and apply **Canny edge detection**
   - Convert to **HLS color space** and threshold the saturation channel
   - Combine edge and color masks into a binary lane image

3. **Sliding Window Search**
   - Compute a column histogram on the lower half of the image
   - Detect left and right lane base positions
   - Use sliding windows to track lane pixels upward
   - Collect lane pixel coordinates

4. **Polynomial Fitting**
   - Fit second-degree polynomials to left and right lane pixels
   - Generate smooth lane curves

5. **Lane Polygon Creation**
   - Construct a polygon between left and right lane boundaries
   - Fill the polygon in bird’s-eye space

6. **Inverse Projection & Overlay**
   - Warp the filled lane polygon back to the original camera view
   - Overlay the detected lane area onto the original frame

---

## Note:
   - The values for src and dst matrices in homography_matrix() function depends on the camera position, so the current values may not produce optimal results for another dashcam. Feel free to play around with the values to get optimal results if another dashcam footage is being used. Commented lines 15-16 are useful if you want to see the specific coordinates you are transforming. The coords image in the data folder shows how the coordinates look for this particular dashcam, you should aim for something very close to this.

## Future Work/Improvements
   1. Implement a depth detection system, where it can identify obstacles and other car in front and calculate how far away they are
   2. Make the lane polygon transparent for better visuals
   3. Use this vision pipeline to implement a fully autonomous robot/car in ROS2
    
