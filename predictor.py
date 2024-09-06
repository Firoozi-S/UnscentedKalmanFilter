from typing import Any

import numpy as np

import commons

BETA_0 = -0.59783
H_0 = 13.406
Gm_0 = 3.9860* (10**5)
R_0 = 6374

class Predictor():
    def __init__(self):
        """"""
        
    
    def DistanceFromEarth(self, position):
        """ Distance from center of earth"""
        return np.sqrt(np.square(position[0]) + np.square(position[1]))
    
    def AbsoluteSpeed(self, velocity):
        """ Absolute vehicle speed"""
        return np.sqrt(np.square(velocity[0]) + np.square(velocity[1]))
    
    def Gravitational(self, position):
        """ Gravitational term"""
        return - Gm_0 / (self.DistanceFromEarth(position)**3)
    
    def BallisticCoefficient(self, aerodynamic_property):
        """ Ballistic coefficient"""
        return BETA_0 * np.exp(aerodynamic_property)
    
    def DragForce(self, position, velocity, aerodynamic_property):
        """ Drag force term"""
        return - self.BallisticCoefficient(aerodynamic_property) * np.exp((R_0 - self.DistanceFromEarth(position)) / H_0) * self.AbsoluteSpeed(velocity)
    
    def __call__(self, state_space):
        """ Given unscented transformed states, returns priori estimate of dynamic state"""
        position = (state_space[0], state_space[1])
        velocity = (state_space[2], state_space[3])
        aerodynamic_property = state_space[4]

        state_dynamic = np.zeros((np.size(state_space), 1))

        state_dynamic[0, :] = state_space[2]

        state_dynamic[1, :] = state_space[3]

        state_dynamic[2, :] = self.DragForce(position, velocity, aerodynamic_property) * state_space[2] + self.Gravitational(position) * state_space[0] + state_space[5]
        
        state_dynamic[3, :] = self.DragForce(position, velocity, aerodynamic_property) * state_space[3] + self.Gravitational(position) * state_space[1] + state_space[6]
        
        state_dynamic[4, :] = state_space[7]

        state_dynamic[5, :] = state_space[5]

        state_dynamic[6, :] = state_space[6]

        state_dynamic[7, :] = state_space[7]

        return state_dynamic
    
        
    