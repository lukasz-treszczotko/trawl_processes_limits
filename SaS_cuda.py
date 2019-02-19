import torch
from math import pi
import numpy as np


class SaS_cuda():
    def __init__(self, alpha):
        self.alpha = alpha
        
    def sample(self, num_samples):
        u_1 = torch.cuda.FloatTensor((num_samples)).uniform_()
        u_2 = torch.cuda.FloatTensor((num_samples)).uniform_()
        gamma = u_1 * pi - pi/2
        w = -torch.log(1-u_2)
        A = torch.sin(self.alpha*gamma)/torch.pow(torch.cos(gamma), 1./self.alpha)
        B = torch.pow(torch.cos(gamma-self.alpha*gamma)/w, (1-self.alpha)/self.alpha)
        return (A*B).cpu().numpy()
