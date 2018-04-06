# ibug_head_pose_estimator
Here is the iBUG head pose estimator. You can use it to estimate head pose in terms of pitch, yaw, and roll from the coordinates of 49 facial landmarks. The algorithm is based on linear regression and rigid 2D alignment, so the code runs quite fasts, which typically takes no more than 1ms to process a frame.

## How to install
1. Install numpy if it's not already there: `pip install numpy` or `conda install -c anaconda numpy`
2. Copy this package to where Python can see. You don't need to copy [./test_data] though unless you want to run the built-in test.

## How to use
Please refer to the test function in [./ibug_head_pose_estimator.py], or just do as follows:
```
from ibug_head_pose_estimator import HeadPoseEstimator
estimator = HeadPoseEstimator()
pitch, yaw, roll = estimator.estimate_head_pose(landmarks) # landmarks must be a 49x2 matrix
```
