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

베이스 코드에서 제공하는 calibration 이미지들(./camera_cal/*.jpg)과 OpenCV의 cv2.findChessboardCorners() 함수로 얻은 `imgpoints`와 `objpoints`를 인자로 cv2.calibrateCamera()를 호출하여 camera matrix와 distortion coefficients를 얻는다. 이렇게 얻은 정보를 이용하여 cv2.undistort()를 호출하면 최종적으로 distortion이 보정된 이미지를 얻을 수 있다. 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

제공된 테스트 이미지에 distortion correction를 수행한 결과는 아래와 같다. 

![alt text][image2]

#### 2. Create a thresholded binary image

차선 영역에서 강하게 나타나는 특징을 threshold 처리하여 일단 binary image로 만들고 나면 그 이후에는 차선 검출을 수월하게 진행할 수 있다. 간단하게는 x, y 방향으로 sobel filter를 적용하여 edge를 잡아내면 검정색 아스팔트에서 노란색과 하얀색 차선을 쉽게 구별할 수 있다. 

![alt text][image3]

그리고 상식적으로 차선이 수평선 보다는 수직선에 가까운 형태로 나타난다는 점을 이용해서 gradient의 방향을 차선의 edge 추출에 활용할 수도 있다. 

![alt text][image4]

위에서처럼 edge 기반으로 접근하게 되면 그림자의 영향이나 아스팔트가 밝은색 계열일 때 검출이 어렵다. 이런 문제를 해결하기 위해 빛의 변화에 덜 민감한 색상 체계를 사용하여 차선 색상에 해당하는 부분을 추출해내는 것도 좋은 방법이 될 수 있다. 여기에서는 HLS color space에서 L과 S 채널을 사용하였다. 

![alt text][image5]

위 3가지 binary image를 모두 combine하여 각각 RGB 채널로 나타내면 아래와 같다. 

![alt text][image6]

#### 3. Perspective transform

베이스 프로젝트에서 제공한 writeup_template을 참고하여 아래와 같이 perspectivetransform을 위한 source와 destination 영역을 지정하였다.

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

Warp한 결과는 아래 이미지에서 확인할 수 있다. 

![alt text][image7]

#### 4. Fit lane-line pixels

thresholded binary image를 Warp하여 왼쪽과 오른쪽에서 나타나는 point들을 각각 왼쪽과 오른쪽 lane line이라고 가정할 수 있다. 두 영역에 속한 point들을 이용해 2차 곡선으로 fitting하면 아래와 같은 결과를 얻을 수 있다. 

![alt text][image8]

#### 5. Calculated the curvature and the vehicle position

x와 y차원에서의 픽셀 당 거리값의 추정값(`ym_per_pix`, `xm_per_pix`)을 이용하여 다음 공식으로 곡률반경을 구한다. ![alt text][image9] 차선으로부터 차 중심의 거리는 단순히 카메라의 위치가 차 중심이라고 가정하고 단순 계산한다. 

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

warp된 상태에서 표시한 lane line을 다시 원래 이미지에 덮어 씌우는 것은 앞에서 수행한 perspective transform의 역변환을 통해 가능하며 아래와 같은 코드로 실행 가능하다. 

```python
Minv = cv2.getPerspectiveTransform(dst, src)
newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
```

![alt text][image10]

---

### Pipeline (video)

#### 1. Final video output

Here's a [link to my video result](https://www.youtube.com/watch?v=Sj5GB-UKWRE)
<div align="left">
  <a href="https://www.youtube.com/watch?v=Sj5GB-UKWRE"><img src="https://img.youtube.com/vi/Sj5GB-UKWRE/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>

---

### Discussion

#### 1. Briefly discuss any problems / issues 

HSL 색상 이미지를 사용했음에도 특정 상황에서는(예를 들어, 밝은색 계통의 도로이면서 그림자 때문에 차선 이외의 곳에서 edge가 강하게 나타나는 경우 등) 그림자의 경계를 차선으로 인식하는 경향이 강하게 나타나는 단점을 발견하였다. 
