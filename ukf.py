import numpy as np

from predictor import Predictor
from corrector import Corrector
from ut import UnscentedTransform

from commons import PROCESS_NOISE_MEAN, PROCESS_NOISE_COV, MEASURMENT_NOISE_COV, MEASURMENT_NOISE_MEAN

class UKF:
    def __init__(self):
        """ Unscented Kalman Filter"""
        self.n_dim = 8
        self.ut = UnscentedTransform()
        self.predictor = Predictor()
        self.corrector = Corrector()

    def CallCorrector(self, state, noise):
        """ Call Sensor Model"""
        return self.corrector.__call__(state, noise)
    
    def CallPredictor(self, state, noise):
        """ Call Presdiction Model"""
        return self.predictor.__call__(state, noise)
    
    def GetUnscentedTransformation(self,
                                   state_mean,
                                   state_cov,
                                   noise_mean,
                                   noise_cov):
        """ Get Unscented Transformation"""
        return self.ut.__call__(state_mean,
                                state_cov,
                                noise_mean,
                                noise_cov)
    
    def GetKappa(self, cov_xy, cov_yy):
        """ Calculate Kappa"""
        return cov_xy @ np.linalg.inv(cov_yy)

    def __call__(self, 
                 initial_mean,
                 initial_cov,
                 ):
        predicted_state = []
        prev_mean = initial_mean
        prev_cov = initial_cov
        
        n_dim = np.size(initial_mean)
        q_dim = np.size(PROCESS_NOISE_MEAN)
        
        next_state = np.zeros((n_dim, 2 * (n_dim + q_dim) + 1))
        m_weight, c_weight, sigma_points = self.GetUnscentedTransformation(prev_mean,
                                                                           prev_cov,
                                                                           PROCESS_NOISE_MEAN,
                                                                           PROCESS_NOISE_COV)
        
        sigma_state = sigma_points[ :n_dim, :]
        sigma_noise = sigma_points[n_dim:, :]

        
        for i in range(np.size(sigma_points, axis = 1)):
            next_state[:, i] = np.squeeze(self.CallPredictor(sigma_state[:,i],
                                                             sigma_noise[:, i]))
                            
        avg_mean_next_state = np.sum(m_weight * next_state, axis = 1, keepdims = True)
        avg_cov_next_state = (c_weight 
                                * (next_state - avg_mean_next_state) 
                                @ np.transpose(next_state - avg_mean_next_state))
        
        m_weight, c_weight, sigma_points = self.GetUnscentedTransformation(np.reshape(avg_mean_next_state, (1,n_dim)),
                                                                           avg_cov_next_state,
                                                                           MEASURMENT_NOISE_MEAN,
                                                                           MEASURMENT_NOISE_COV)
        
        state_sigma = sigma_points[:n_dim, :]
        noise_sigma = sigma_points[n_dim:, :]
    
        a_dim = 2 * (n_dim + np.size(MEASURMENT_NOISE_MEAN)) + 1
        z_state = np.zeros((2, a_dim))

        for i in range(np.size(sigma_points, axis = 1)):
            z_state[:, i]= self.CallCorrector(state_sigma[:, i],
                                              noise_sigma[:, i])
            
        z_mean = np.sum(m_weight * z_state, 
                        axis = 1, 
                        keepdims = True)

        z_var = z_state - z_mean
        z_cov_yy = (c_weight * z_var 
                    @ np.transpose(z_var))
        
        z_cov_xy = (c_weight * 
                    (sigma_points[:n_dim,:]
                     - np.sum(sigma_points[:n_dim,:], axis = 1, keepdims = True)) 
                    @ np.transpose(z_var))
                    
        kappa = self.GetKappa(z_cov_xy, z_cov_yy)


        avg_noise = np.sum(noise_sigma, axis = 1)
        avg_mean_next_state = np.reshape(avg_mean_next_state, (5,))
        estimated_measurement = self.CallCorrector(avg_mean_next_state,
                                                   avg_noise)
        
        residual = np.transpose(np.transpose(z_mean) - estimated_measurement)
        corrected_state_mean = (np.expand_dims(avg_mean_next_state, 1) + 
                                kappa @ 
                                residual)
        
        corrected_state_cov = avg_cov_next_state - kappa @ np.transpose(z_cov_xy)

        predicted_state.append((corrected_state_mean, corrected_state_cov))

        prev_mean = np.transpose(corrected_state_mean)
        prev_cov = corrected_state_cov
        
            
        return predicted_state
 



    
        