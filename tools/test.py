from models.QuadNet import QuadNet
import torch
import torch.nn.functional as F
import numpy as np
from dataUtils import MyDataSet as Dataset, DeviceDataLoader as DDL


def test():
    model = get_pretrained()

    # get parabola
    # a = float(input("enter a:"))
    # b = float(input("enter b:"))
    # c = float(input("enter c:"))
    # a = 1.
    # b = 36.
    # c = -500.
    # par_l = [a, b, c]
    # root_l = np.roots(par_l)
    # if np.isreal(root_l[0]) and np.isreal(root_l[1]):
    #     print("there are 2 roots")
    # elif np.isreal(root_l[0]) or np.isreal(root_l[1]):
    #     print("there is 1 root")
    # else:
    #     print("there are no roots")
    #
    # # build tensor for QuadNet
    # inputs = torch.FloatTensor(par_l)
    #
    # # get roots from net
    # print("Network prediction is:", F.softmax(inputs).argmax(), "roots")
    dataset = Dataset.Quadset()

    # Initialize data loaders
    dl = torch.utils.data.DataLoader(dataset,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=False
                                           )

    model.eval()
    for parameters, labels in dl:
        print("Model output: ", model(parameters), "Labels: ", labels)


def get_pretrained():  # initialize and load
    model = QuadNet(input_size=3,
                    output_size=3,
                    )
    device = DDL.get_default_device()
    model.load_state_dict(torch.load("QuadNet_ep6.pth", map_location=device))
    return model

if __name__ == "__main__":
    test()
