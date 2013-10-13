#!/usr/bin/env/python
import numpy as np
import cv2


class KalmanPrototype():
    state_transition = np.array([2,2])
    transition_cov = np.array([2,2])
    measurement_cov = np.array([1,1])
    measurement_matrix = np.array([1,2])    

    #needs better name 
    def filter(measurement, prev_state, prev_cov, self): 
        pred_state, pred_cov = predict(prev_state, prev_cov)
        kalman_state, kalman_cov = measure(pred_state, pred_cov, measurement)
        return kalman_state, kalman_cov
        
#####
    def predict(prev_state, prev_cov, self):
        pred_state = np.dot(state_transition, prev_state)
        pred_cov = np.dot(np.dot(state_transition, prev_cov), 
                            state_transition.transpose())
        pred_cov += transition_cov
        return pred_state, pred_cov
        

    def measure(pred_state, pred_cov, measurement, self):
        scalar_gain = np.dot(measurement_matrix, np.dot(pred_cov,
                              measurement_matrix.transpose())) + measurement_cov
        scalar_gain = 1/(scalar_gain)
        kalman_gain = np.dot(pred_cov, measurement_matrix.transpose())
        kalman_gain *= scalar_gain

        kalman_state = pred_state + kalman_gain(measurement -
                              np.dot(measurement_matrix, pred_state))
        kalman_cov = np.eye(2, 2) - np.dot(kalman_gain, measurement_matrix)
        kalman_cov = np.dot(kalman_cov, pred_cov)
        return kalman_state, kalman_cov
    
    


    prev_state = np.array([2,1])
    prev_cov = np.array([2,2])
