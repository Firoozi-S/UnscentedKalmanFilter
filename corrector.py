import numpy as np

import math

X_0 = 6375
Y_0 = 0

class Corrector():
    def __init__(self):
        """ Radar Initilization"""
        
    def Range(self, position, radius_noise):
        """ Radar Range Calculation"""
        return np.sqrt(np.square(position[0] - X_0) + np.square(position[1] - Y_0)) + radius_noise
    
    def Bearing(self, position, angle_noise):
        """ Radar Angle of Arrival"""
        return math.degrees(math.atan((position[1] - Y_0) / (position[0] - X_0)) + angle_noise)
    
    def __call__(self, state_space, noise_space):
        """ Calculate Distance and AoA of target object"""

        state_position = state_space[0], state_space[1]

        radius_noise = noise_space[0]
        angle_noise = noise_space[1]

        radius = self.Range(state_position, radius_noise)
        theta = self.Bearing(state_position, angle_noise)

        return (radius, theta)