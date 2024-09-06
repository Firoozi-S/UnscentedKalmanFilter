import numpy as np

from typing import Any

from commons import INITIAL_COV, PROCESS_NOISE_COV, INITIAL_STATE, PROCESS_NOISE_MEAN

ALPHA = 10**-3
BETA = 2
KAPPA = 0

def make_positive_definite(cov, epsilon = 1e-10):
    """ Ensuring covariance is positive definite """
    return cov + np.eye(cov.shape[0]) * epsilon


class UnscentedTransform:
    def __init__(self):

        self.n_dim = 0

        return
    
    def ComputeSigmaPoints(self,
                           mean,
                           cov):
        
        mean = np.squeeze(mean)

        sigma_points = np.zeros((np.size(mean), 2 * self.n_dim + 1) )    
                
        sigma_points[:, 0] = mean

        cov = make_positive_definite(cov)
        sqrt_cov = np.linalg.cholesky((self.n_dim + KAPPA) * cov)

        for i in range(self.n_dim):
            sigma_points[: , i + 1] = mean + sqrt_cov[:, i]
            sigma_points[:, self.n_dim + 1 + i] = mean - sqrt_cov[:, i]
  
        return sigma_points
    
    def ComputeWeights(self):
        m_weight = np.zeros((1, 2 * self.n_dim + 1))
        c_weight = np.zeros((1, 2 * self.n_dim + 1))

        m_weight[:,0] = KAPPA / (self.n_dim + KAPPA) 

        c_weight[:, 0] = m_weight[:,0] + (1 - np.square(ALPHA) + BETA)

        for i in range(1, 2* self.n_dim +1):
            c_weight[:, i] = (0.5) / (self.n_dim + KAPPA) 
            m_weight[:, i] = (0.5) / (self.n_dim + KAPPA) 

        return m_weight, c_weight
    

    def ComputeAugmentedCov(self, state_cov, noise_cov):
        """ Compute covariance matrix of state and noise"""

        cross_cov = np.zeros((np.size(state_cov[0]),np.size(noise_cov[0])))

        augmented_cov_top = np.hstack([state_cov, cross_cov])
        augmented_cov_bottom = np.hstack([cross_cov.T, noise_cov])

        augmented_cov = np.vstack([augmented_cov_top, augmented_cov_bottom])

        return augmented_cov   
    
    def ComputeAugmentedMean(self, state_mean, noise_mean):
        """ Compute mean matrix of state and noise"""

        return np.vstack([state_mean.T, noise_mean.T])
     
    def __call__(self,
                 state_mean,
                 state_cov,
                 noise_mean,
                 noise_cov):
        """ Calculate Sigma Points and Weights"""
        cov = self.ComputeAugmentedCov(state_cov, noise_cov)
        mean = self.ComputeAugmentedMean(state_mean, noise_mean)

        self.n_dim = np.size(mean)

        sigma_points = self.ComputeSigmaPoints(mean, cov)

        m_weight, c_weight = self.ComputeWeights()

        return m_weight, c_weight, sigma_points






