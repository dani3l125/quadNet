import torch
import torch.nn as nn
import tensorly as tl
from tensorly.tenalg import khatri_rao as khart
from dataUtils.DeviceDataLoader import get_default_device


# TODO, get input pulynomials instead
class GDS(nn.Module):
    def __init__(self, degree, dim, neurons=1, rand=False):
        super(GDS, self).__init__()
        tl.set_backend('pytorch')  # taking derivative of khatri-rao.
        self.device = get_default_device()
        self.dim = dim
        self.degrees = range(1, degree + 1)
        self.neurons = nn.ParameterList([nn.Parameter(torch.rand
                                                      (dim, dtype=torch.float64, requires_grad=True,
                                                       device=self.device)) for i in range(neurons)])
        self.polynomials = []
        for i in range(dim):
            polynomial = [torch.rand(1, dtype=torch.float64, requires_grad=False, device=self.device) if rand else
                          torch.zeros(1, dtype=torch.float64, requires_grad=False, device=self.device)]
            cols = 1
            for n in self.degrees:
                cols *= degree
                polynomial.append(
                    torch.rand(cols, dtype=torch.float64, requires_grad=False, device=self.device)
                    if rand else
                    torch.zeros(cols, dtype=torch.float64, requires_grad=False, device=self.device)
                )
            self.polynomials.append(polynomial)
        self.solution = (False, -1)
        self.loss = nn.L1Loss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.neurons, lr=0.1)

    def forward(self):
        output = torch.zeros(len(self.neurons), len(self.polynomials), device=self.device)
        for n_idx, neuron in enumerate(self.neurons):
            for p_idx, pol in enumerate(self.polynomials):
                neuron = self.neurons[n_idx].T
                output[n_idx, p_idx] = pol[0]
                for n in self.degrees:
                    output[n_idx, p_idx] += torch.sum((pol[n] * neuron.T))
                    neuron = khart([self.neurons[n_idx].T, neuron])
        best_v = torch.min(torch.max(output.square(), dim=1).values)
        if best_v <= 0.00001:
            a = 1
        return output

    def insert_coefficients(self):  # TODO: leahlil
        # print("polynomial 1:")
        # print("enter x1x1")
        # self.polynomials[0][2][0] = float(input())
        # print("enter x1x2")
        # self.polynomials[0][2][2] = float(input())
        # print("enter x2x2")
        # self.polynomials[0][2][3] = float(input())
        # print("enter bias")
        # self.polynomials[0][0][0] = float(input())
        # print("enter x1")
        # self.polynomials[0][1][0] = float(input())
        # print("enter x2")
        # self.polynomials[0][1][1] = float(input())
        # print("polynomial 2:")
        # print("enter x1x1")
        # self.polynomials[1][2][0] = float(input())
        # print("enter x1x2")
        # self.polynomials[1][2][2] = float(input())
        # print("enter x2x2")
        # self.polynomials[1][2][3] = float(input())
        # print("enter bias")
        # self.polynomials[1][0][0] = float(input())
        # print("enter x1")
        # self.polynomials[1][1][0] = float(input())
        # print("enter x2")
        # self.polynomials[1][1][1] = float(input())
        print("polynomial 1:")
        print("enter x1x1")
        self.polynomials[0][2][0] = 1
        print("enter x1x2")
        self.polynomials[0][2][1] = -2
        print("enter x2x2")
        self.polynomials[0][2][3] = 2
        print("enter bias")
        self.polynomials[0][0][0] = -1
        print("enter x1")
        self.polynomials[0][1][0] = 1
        print("enter x2")
        self.polynomials[0][1][1] = -1
        print("polynomial 2:")
        print("enter x1x1")
        self.polynomials[1][2][0] = -1
        print("enter x1x2")
        self.polynomials[1][2][1] = -1
        print("enter x2x2")
        self.polynomials[1][2][3] = 2
        print("enter bias")
        self.polynomials[1][0][0] = 0
        print("enter x1")
        self.polynomials[1][1][0] = 1
        print("enter x2")
        self.polynomials[1][1][1] = 0

    def solve(self):
        while not self.solution[0]:
            output = torch.max(self().square(), dim=1)
            output = output.values
            loss = self.loss(output, torch.zeros(len(self.neurons), device=self.device))
            best_i = torch.argmin(output)
            best_v = output[best_i]
            print(f"loss: {loss}, best: {best_v}")
            print(f"best{self.neurons[best_i]}")
            if best_v <= 0.00001:
                self.solution = (True, best_i)
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.neurons[self.solution[1]]


if __name__ == "__main__":
    tmp = GDS(2, 2, 1000)
    print("input:{}".format(tmp.neurons[0]))
    tmp.insert_coefficients()
    for i, pol in enumerate(tmp.polynomials):
        print("{}th coefficients:{}".format(i, tmp.polynomials[i]))
    print("Solution:{}".format(tmp.solve()))
