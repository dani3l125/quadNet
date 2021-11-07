from models.QuadNet import QuadNet
import torch
import numpy as np


def test():
    model = get_pretrained()

    # get parabola
    # a = float(input("enter a:"))
    # b = float(input("enter b:"))
    # c = float(input("enter c:"))
    a = 1.
    b = -333.
    c = 765.
    par_l = [a, b, c]
    root_l = np.roots(par_l)
    print("roots are:", root_l)

    # build tensor for QuadNet
    inputs = torch.FloatTensor(par_l)

    # get roots from net
    print("Network roots are:", model(inputs))

def get_pretrained():  # initialize and load
    model = QuadNet(input_size=3,
                    output_size=4,
                    )
    model.load_state_dict(torch.load("QuadNet.pth"))
    return model

if __name__ == "__main__":
    test()
