import tensorly as tl
import torch
import torch.nn as nn

class PiNet(nn.module):
    def __init__(self, degree, in_size, out_size):
        super().__init__()
        self.degree = degree
        self.in_size = in_size
        self.out_size = out_size
        self.weights_matrices = []
        cols = 1
        for n in range(degree):
            cols *= in_size
            self.weights_matrices.append(torch.rand((out_size, cols), requires_grad=True))

