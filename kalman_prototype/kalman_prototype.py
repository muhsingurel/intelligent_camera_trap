##################################
# Kalman Filter 360 Camera Test
# 
# Runs a Kalman Filter using an angular position, angular velocity state 
# transition model. Red Dot shows unfiltered angular position estimate, green 
# dot shows kalman estimate.
# ################################
import numpy as np
import pylab as pl
from pykalman import KalmanFilter
import cv2
import math

cv2.namedWindow("BG MoG")
cv2.namedWindow("Color")

cap = cv2.VideoCapture(1)
bg_mog = cv2.BackgroundSubtractorMOG()

# specify parameters
random_state = np.random.RandomState(0)
transition_matrix = [[1, 0], 
                     [1, 1]]

transition_offset = [0, 0]
observation_matrix = [[1, 0], [0, 0]] 
observation_offset = [0, 0]
transition_covariance = np.eye(2)
observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
initial_state_mean = [90, 0]
initial_state_covariance = [[1, 0.1], [-0.1, 1]]

# build model
kf = KalmanFilter(
    transition_matrix, observation_matrix, transition_covariance,
    observation_covariance, transition_offset, observation_offset,
    initial_state_mean, initial_state_covariance,
    random_state=random_state
)

got_frame, img = cap.read()

theta_prev = 90
kalman_prev = initial_state_mean
kalman_cov = np.eye(2)

while (got_frame == True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = bg_mog.apply(img, None, 0.01)

    contours, hierarchy = cv2.findContours(img_thresh, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Use largest blob
    area = np.zeros(len(contours))
    for i in range(0, len(contours)):
        area[i] = (cv2.contourArea(contours[i]))

    if len(area > 0):
        max_contour = contours[area.argmax()]
    
        moments = cv2.moments(max_contour)
        m00 = moments['m00']
        m01 = moments['m01']
        m10 = moments['m10']

        if m00 > 0:
            x_pos = int((m10 / m00)) - 310
            y_pos = int((m01 / m00)) - 255

            theta_rad = math.atan2(y_pos, x_pos)
            theta_deg = theta_rad * 180 / (math.pi)
            w_deg = (theta_deg - theta_prev)
            theta_prev = theta_deg
            
            r = 50
            x_pos = int(r * math.cos(theta_rad))
            y_pos = int(r * math.sin(theta_rad))
            cv2.circle(img, (x_pos + 310, y_pos + 255), 4, (0, 0, 255), -1)
            state_est, cov_est = kf.filter_update(kalman_prev, kalman_cov, 
                                           [theta_deg, w_deg])
            kalman_prev = state_est
            kalman_cov = cov_est
            kalman_rad = state_est[0] * (math.pi / 180)
            kalman_x = int(r * math.cos(kalman_rad))
            kalman_y = int(r * math.sin(kalman_rad))
            
            cv2.circle(img, (kalman_x + 310, kalman_y + 255), 4, (0, 255, 0), -1)
        

    cv2.imshow("Color", img)
    cv2.imshow("BG MoG", img_thresh)

    cv2.waitKey(1)

    got_frame, img = cap.read()

