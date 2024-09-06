import numpy as np

import math

import commons

X_0 = 6375
Y_0 = 0

NOISE_1_VARIANCE = 1
NOISE_2_VARIANCE = 17 * (10**-3)

class Corrector():
    def __init__(self):
        """"""
        self.measurement_noise_1 = np.random.normal(loc = 0, scale = NOISE_1_VARIANCE, size = np.size(commons.K))
        self.measurement_noise_2 = np.random.normal(loc = 0, scale = NOISE_2_VARIANCE, size = np.size(commons.K))
        
    def Range(self, position, k):
        """"""
        return np.sqrt(np.square(position[0] - X_0) + np.square(position[1] - Y_0)) + self.measurement_noise_1[k]
    
    def Bearing(self, position, k):
        """"""
        return math.degrees(math.atan((position[1] - Y_0) / (position[0] - X_0)) + self.measurement_noise_2[k])
    

    
    def __call__(self, state_space, k):

        state_position = state_space[0], state_space[1]

        radius = self.Range(state_position, k)
        theta = self.Bearing(state_position, k)


        return (radius, theta)