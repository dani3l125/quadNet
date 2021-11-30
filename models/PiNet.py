import torch
import torch.nn as nn
from time import time
import tensorly as tl
from tensorly.tenalg import khatri_rao as khart


# from scipy.linalg import khatri_rao as khart


class PiNet(nn.Module):
    def __init__(self, degree, in_size, out_size):
        tl.set_backend('pytorch')
        super().__init__()
        self.degrees = range(1, degree + 1)
        self.in_size = in_size
        self.loss = nn.L1Loss(reduction='sum')
        self.out_size = out_size
        self.weights_matrices = [nn.Parameter(torch.rand(out_size, 1, dtype=torch.float64, requires_grad=True))]
        cols = 1
        for n in self.degrees:
            cols *= in_size
            self.weights_matrices.append(
                nn.Parameter(torch.rand(out_size, cols, dtype=torch.float64, requires_grad=True)))
            # self.weights_matrices.append(nn.Linear(out_size, cols, bias=False))
        self.weights_matrices = nn.ParameterList(self.weights_matrices)

    def forward(self, input):

        z_n = input.T
        out = torch.matmul(self.weights_matrices[0], torch.ones(1, input.size()[0], dtype=torch.float64, device='cuda'))
        for n in self.degrees:
            out = torch.add(out, self.weights_matrices[n].mm(z_n))
            # out = torch.add(out, self.weights_matrices[n].mm(z_n))
            z_n = khart([input.T, z_n])
        return out.T

    def get_losses(self, batch):
        parameters, values = batch
        values = values.flatten()
        out = self(parameters).flatten()
        return self.loss(out, values)

    def training_step(self, batch):
        return self.get_losses(batch)

    def validation_step(self, batch):
        loss = 1 / batch[0].size()[0] * self.get_losses(batch)
        # return {'mse': loss.detach(), 'accuracy': self.accuracy(out, values)}
        return {'mse': loss.detach()}

    def validation_epoch_end(self, outputs_val):
        batch_losses_val = [x['mse'] for x in outputs_val]
        # batch_acc_val = [x['accuracy'] for x in outputs_val]
        epoch_loss_val = torch.stack(batch_losses_val).mean()  # Combine losses
        # epoch_acc_val = torch.stack(batch_acc_val).mean()  # Combine accuracies
        # return {'val_mse': epoch_loss_val.item(),
        #         'val_acc': epoch_acc_val.item()}
        return {'val_mse': epoch_loss_val.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], validation mean squared error: {:.6f}".format(
            epoch, result['val_mse']))

    def accuracy(self, outputs, values):
        print("predictions: ", outputs, "values: ", values)
        # return torch.tensor(torch.sum(values[0]-1 <= outputs[0] <= values[0]+1).item() / len(preds))
        return 100
