import tensorly as tl
import torch
import torch.nn as nn

class PiNet(nn.module):
    def __init__(self, degree, in_size, out_size):
        super().__init__()
        self.degrees = range(1, degree+1)
        self.in_size = in_size
        self.out_size = out_size
        self.weights_matrices = []
        self.bias_matrices = torch.rand(out_size, 1)
        cols = 1
        for n in self.degrees:
            cols *= in_size
            self.weights_matrices.append(nn.Linear(out_size, cols, bias=false))

    def forward(self, input):
        z_n = input
        out = self.biases
        for n in self.degrees:
            out += self.weights_matrices[i](z_n)
            z_n = torch.kron(z_n, input)




