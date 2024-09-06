import numpy as np

from predictor import Predictor
from corrector import Corrector
from ut import UnscentedTransform

class UKF:
    def __init__(self):
        self.ut = UnscentedTransform()
        self.predictor = Predictor()
        self.corrector = Corrector()

    def ComputeCrossCov(self):
        return

    def InitState(self):
        return

    def __call__(self):
        return
 






    
        