## Writeup Template

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_calibration.png "Calibration"
[image2]: ./output_images/corrected_image.png "Correct"
[image3]: ./output_images/grad_xy.png "XY Grad"
[image4]: ./output_images/grad_dir.png "Dir Grad"
[image5]: ./output_images/hls.png "HLS"
[image6]: ./output_images/pipeline_result.png "Pipeline"
[image7]: ./output_images/perspective_transform.png "Perspective"
[image8]: ./output_images/left_right_lane_line.png "Lane Line"
[image9]: ./output_images/curvature.png "Curvature"
[image10]: ./output_images/warp_back.png "Fit Visual"
[image11]: ./output_images/final_result.png "Output"
[video1]: ./project_video_result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README 

You're reading it!

### Camera Calibration

#### 1. Computed the camera matrix and distortion coefficients

We call cv2.findChessboardCorners() and cv2.calibrateCamera() function with the parameters `imgpoints` and `objpoints` provided from the template project(./camera_cal/*.jpg), and then we can obtain the camera matrix and distortion coefficients. Using this information, you can call cv2.undistort() to get the final distortion corrected image.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The results of the distortion correction on the test image is as follows.

![alt text][image2]

#### 2. Create a thresholded binary image

Once the features that appear strong in the lane area is thresholded and made into a binary image, the lane detection can proceed easily using Sobel filter.

![alt text][image3]

Also, it is possible to use the gradient direction to detect the lane lines by using the fact that the lane line is more likely vertical than horizontal in common sense.

![alt text][image4]

As shown above, edge-based approaches make it difficult to detect lane lines when the effect of shadows or luminance variation is strong. To solve this problem, it may be a good idea to use a color scheme that is less sensitive to changes in light. Here, the L and S channels of the HLS color space, the L channel of the LUV, and the B channel of the LAB are used in combination.

![alt text][image5]

Combine all of the above binary images into RGB channels.

![alt text][image6]

#### 3. Perspective transform

The source and destination regions for the perspective transform as shown below. (refer to the [writeup_template.md of the template project](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md))

```python
img_size = result.shape[1::-1]

src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 203.3, 720    | 320, 720      |
| 1126.7, 720   | 960, 720      |
| 695, 460      | 960, 0        |

Warped image is as follows.

![alt text][image7]

#### 4. Fit lane-line pixels

In the warped image of the thresholded binary image, we can consider the left and right part of the points as left and right lane lines respectively. Fitting the points with a quadratic curve produces the following results.

![alt text][image8]

#### 5. Calculated the curvature and the vehicle position

Using the estimated value of the real distance per pixel in the image, the radius of curvature is obtained by the following formula.

![alt text][image9] 

The distance from the lane line to the center of the vehicle is calculated simply by assuming that the position of the camera is the center of the car.

```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

# Calculate the new radii of curvature (실좌표계에서의 curvature)
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

# Now our radius of curvature is in meters
print('curvature (real coord):', left_curverad, 'm', right_curverad, 'm')

# 좌/우 lane line의 시작점 x좌표
left_fit_pos = left_fitx[-1]
right_fit_pos = right_fitx[-1]
print('left/right x position:', left_fit_pos, right_fit_pos)
```

#### 6. Warp back 

Warping back the lane line to the original image is possible through the inverse transformation of the previous perspective transform, and can be simply implemented with the following code.

```python
Minv = cv2.getPerspectiveTransform(dst, src)
newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
```

![alt text][image10]

---

### Pipeline (video)

#### 1. Final video output

We added the 'Reset' and 'Smoothing' code to make the final result more robust and natural when applying the image processing pipeline to the video. If the sanity check fails for 5 consecutive frames, the existing information is initialized and a fitting line is computed from scratch.

```python
if left_lane_line.best_fit is None or right_lane_line.best_fit is None:
        left_fit, right_fit = fit_lines_from_scratch(binary_warped)
    else :
        left_fit, right_fit = fit_lines_with_previous_lines(binary_warped, 
                                                    left_lane_line.best_fit, 
                                                    right_lane_line.best_fit)
```

Using a moving average of fitting line coefficient for 5 frames, the effect of smoothing is obtained even if the fitting line is occasionally not found in some frames.

```python
left_lane_line.best_fit * 0.8 + left_lane_line.current_fit * 0.2
right_lane_line.best_fit * 0.8 + right_lane_line.current_fit * 0.2
```

Here's a [link to my video result](https://www.youtube.com/watch?v=g2R47Rjs-3Y)
<div align="left">
  <a href="https://www.youtube.com/watch?v=g2R47Rjs-3Y"><img src="https://img.youtube.com/vi/g2R47Rjs-3Y/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>

---

### Discussion

#### 1. Briefly discuss any problems / issues 

Even in the case of using all HSL, LUV and LAB color channels, I found the tendency to recognize the boundaries of shadows as lane lines at the bright-colored pavement.


