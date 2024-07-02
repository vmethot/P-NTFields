import math
import random
import time

import matplotlib.pylab as plt
import numpy as np
import pickle5 as pickle
import torch
from torch import Tensor
from torch.autograd import Variable, grad
from torch.nn import Linear

from models import data_multi as db

torch.backends.cudnn.benchmark = True

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def sigmoid(input):

    return torch.sigmoid(10*input)

class Sigmoid(torch.nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self, input):

        return sigmoid(input)

class DSigmoid(torch.nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self, input):

        return 10*sigmoid(input)*(1-sigmoid(input))

def sigmoid_out(input):

    return torch.sigmoid(0.1*input)

class Sigmoid_out(torch.nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self, input):

        return sigmoid_out(input)

class DSigmoid_out(torch.nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self, input):

        return 0.1*sigmoid_out(input)*(1-sigmoid_out(input))

class DDSigmoid_out(torch.nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self, input):

        return 0.01*sigmoid_out(input)*(1-sigmoid_out(input))*(1-2*sigmoid_out(input))

class NN(torch.nn.Module):

    def __init__(self, device, dim):#10
        super(NN, self).__init__()
        self.dim = dim
        h_size = 128 #512,256
        self.input_size = 128
        self.scale = 10

        self.act = torch.nn.Softplus(beta=self.scale)#ELU,CELU

        # self.env_act = torch.nn.Sigmoid()#ELU
        self.dact = Sigmoid()
        self.ddact = DSigmoid()

        self.actout = Sigmoid_out()#ELU,CELU

        # self.env_act = torch.nn.Sigmoid()#ELU
        self.dactout = DSigmoid_out()
        self.ddactout = DDSigmoid_out()

        self.nl1=3
        self.nl2=3

        self.encoder = torch.nn.ModuleList()
        self.encoder1 = torch.nn.ModuleList()
        # self.encoder.append(Linear(self.dim,h_size))

        self.encoder.append(Linear(2*h_size, h_size))
        self.encoder1.append(Linear(2*h_size, h_size))

        for i in range(self.nl1-1):
            self.encoder.append(Linear(h_size, h_size))
            self.encoder1.append(Linear(h_size, h_size))

        self.encoder.append(Linear(h_size, h_size))

        self.generator = torch.nn.ModuleList()
        self.generator1 = torch.nn.ModuleList()
        for i in range(self.nl2):
            self.generator.append(Linear(2*h_size, 2*h_size))
            self.generator1.append(Linear(2*h_size, 2*h_size))

        self.generator.append(Linear(2*h_size,h_size))
        self.generator.append(Linear(h_size,1))

    def init_weights(self, m):

        if type(m) == torch.nn.Linear:
            stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2
            m.weight.data.uniform_(-stdv, stdv)
            m.bias.data.uniform_(-stdv, stdv)
            # torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def input_mapping(self, x, B):
        w = 2.*np.pi*B
        x_proj = x @ w
        # x_proj = (2.*np.pi*x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)    #  2*len(B)

    def input_mapping_grad(self, x, B):
        w = 2.*np.pi*B
        x_proj = x @ w
        x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        dx =  torch.cat([w *torch.cos(x_proj), -w *torch.sin(x_proj)], dim=-1)
        return x, dx

    def input_mapping_laplace(self, x, B):
        w = 2.0 * np.pi * B
        w = w.unsqueeze(1)
        x_proj = x @ w
        x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        dx =  torch.cat([w *torch.cos(x_proj), -w *torch.sin(x_proj)], dim=-1)
        lx =  torch.cat([-w * w *torch.sin(x_proj), -w * w *torch.cos(x_proj)], dim=-1)
        return x, dx, lx

    def out(self, coords, B):

        coords = coords.clone().detach().requires_grad_(True)
        size = coords.shape[0]
        x0 = coords[:, : self.dim]
        x1 = coords[:, self.dim :]

        x = torch.cat((x0,x1),dim=0)

        x = x.unsqueeze(1)

        x = self.input_mapping(x, B)
        x  = self.act(self.encoder[0](x))
        for ii in range(1, self.nl1):
            x_tmp = x
            x  = self.act(self.encoder[ii](x))
            x  = self.act(self.encoder1[ii](x) + x_tmp)

        x = self.encoder[-1](x)

        x0 = x[:size,...]
        x1 = x[size:,...]

        xx = torch.cat((x0, x1), dim=1)

        x_0 = torch.logsumexp(self.scale * xx, 1) / self.scale
        x_1 = -torch.logsumexp(-self.scale * xx, 1) / self.scale

        x = torch.cat((x_0, x_1),1)

        for ii in range(self.nl2):
            x_tmp = x
            x = self.act(self.generator[ii](x))
            x = self.act(self.generator1[ii](x) + x_tmp)
        y = self.generator[-2](x)
        x = self.act(y)
        y = self.generator[-1](x)
        x = self.actout(y)
        return x, coords

    def init_grad(self, x, w, b):
        y = x@w.T+b
        x = self.act(y)
        dact = self.dact(y)
        dx = w.T * dact
        del y, w, b, dact
        return x, dx

    def linear_grad(self, x, dx, w, b):
        y = x @ w.T + b
        x = y
        dxw = dx @ w.T
        dx = dxw
        del y, w, b, dxw
        return x, dx

    def act_grad(self, x, dx):
        dact = self.dact(x)
        dx = dx * dact
        x = self.act(x)
        del dact
        return x, dx

    def actout_grad(self, x, dx):
        actout = self.actout(x)
        dactout = 0.1 * actout * (1 - actout)
        dx = dx * dactout
        x = actout
        del actout, dactout
        return x, dx

    def out_grad(self, coords, B):
        size = coords.shape[0]
        x0 = coords[:,:self.dim]
        x1 = coords[:, self.dim :]
        x = torch.cat((x0, x1), dim=0).unsqueeze(1)
        x, dx = self.input_mapping_grad(x, B)
        w = self.encoder[0].weight
        b = self.encoder[0].bias
        x, dx = self.linear_grad(x, dx, w, b)
        x, dx = self.act_grad(x, dx)
        for ii in range(1,self.nl1):
            x_tmp, dx_tmp = x, dx

            w = self.encoder[ii].weight
            b = self.encoder[ii].bias
            x, dx = self.linear_grad(x, dx, w, b)
            x, dx = self.act_grad(x, dx)
            w = self.encoder1[ii].weight
            b = self.encoder1[ii].bias
            x, dx = self.linear_grad(x, dx, w, b)
            x, dx = x + x_tmp, dx + dx_tmp
            x, dx = self.act_grad(x, dx)

        w = self.encoder[-1].weight
        b = self.encoder[-1].bias
        x, dx = self.linear_grad(x, dx, w, b)

        x0 = x[:size, ...]
        x1 = x[size:, ...]
        dx0 = dx[:size, ...]
        dx1 = dx[size:, ...]
        x0_1 = x0 - x1
        s0 = self.dact(x0_1)
        s1 = 1 - s0
        dx00 = dx0 * s0
        dx01 = dx1 * s1
        dx10 = dx0 * s1
        dx11 = dx1 * s0
        dx0 = torch.cat((dx00, dx01), dim=1)
        dx1 = torch.cat((dx10, dx11), dim=1)
        dx = torch.cat((dx0, dx1), dim=2)
        xx = torch.cat((x0, x1), dim=1)
        x_0 = torch.logsumexp(self.scale * xx, 1) / self.scale
        x_1 = -torch.logsumexp(-self.scale * xx, 1) / self.scale
        x = torch.cat((x_0, x_1), 1).unsqueeze(1)
        for ii in range(self.nl2):
            x_tmp, dx_tmp = x, dx
            w = self.generator[ii].weight
            b = self.generator[ii].bias
            x, dx = self.linear_grad(x, dx, w, b)
            x, dx = self.act_grad(x, dx)
            w = self.generator1[ii].weight
            b = self.generator1[ii].bias
            x, dx = self.linear_grad(x, dx, w, b)
            x, dx = x+ x_tmp, dx+dx_tmp
            x, dx = self.act_grad(x, dx)
        w = self.generator[-2].weight
        b = self.generator[-2].bias
        x, dx = self.linear_grad(x, dx, w, b)
        x, dx = self.act_grad(x, dx)
        w = self.generator[-1].weight
        b = self.generator[-1].bias
        x, dx = self.linear_grad(x, dx, w, b)
        x, dx = self.actout_grad(x, dx)
        x = x.squeeze(2)
        dx = dx.squeeze(2)

        return x, dx, coords

    def out_backgrad(self, coords, B):
        size = coords.shape[0]
        x0 = coords[:, : self.dim]
        x1 = coords[:, self.dim :]
        x = torch.vstack((x0, x1))
        w_list0 = []
        dact_list0 = []
        w = 2.0 * np.pi * B
        x_proj = x @ w
        x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)
        dact = torch.cat([torch.cos(x_proj), -torch.sin(x_proj)], dim=1)
        w_list0.append(torch.cat((w,w),dim=1))
        dact_list0.append(dact.transpose(0, 1))
        w = self.encoder[0].weight
        b = self.encoder[0].bias
        y = x @ w.T + b
        x = y
        w_list0.append(w.T)
        del y, w, b
        x = self.act(x)
        dact = self.dact(x)
        dact_list0.append(dact.transpose(0, 1))
        del dact
        for ii in range(1, self.nl1):
            x_tmp = x
            w = self.encoder[ii].weight
            b = self.encoder[ii].bias
            y = x @ w.T + b
            x = y
            w_list0.append(w.T)
            del y, w, b
            dact = self.dact(x)
            dact_list0.append(dact.transpose(0, 1))
            x = self.act(x)
            del dact
            w = self.encoder1[ii].weight
            b = self.encoder1[ii].bias
            y = x @ w.T + b
            x = y
            w_list0.append(w.T)
            del y, w, b
            x = x + x_tmp
            dact = self.dact(x)
            dact_list0.append(dact.transpose(0, 1))
            x  = self.act(x)
            del dact
        w = self.encoder[-1].weight
        b = self.encoder[-1].bias
        y = x @ w.T + b
        x = y
        w_list0.append(w.T)
        del y, w, b
        x0 = x[:size,...]
        x1 = x[size:, ...]
        x0_1=x0-x1
        s0 = self.dact(x0_1).transpose(0, 1)
        s1 = 1 - s0
        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        xx = torch.cat((x0, x1), dim=1)
        x_0 = torch.logsumexp(self.scale*xx, 1)/self.scale
        x_1 = -torch.logsumexp(-self.scale * xx, 1) / self.scale
        x = torch.cat((x_0, x_1), 1)
        w_list1 = []
        dact_list1 = []
        for ii in range(self.nl2):
            x_tmp = x
            w = self.generator[ii].weight
            b = self.generator[ii].bias
            y = x @ w.T + b
            x = y
            w_list1.append(w.T)
            del y, w, b
            dact = self.dact(x)
            dact_list1.append(dact.transpose(0, 1))
            x  = self.act(x)
            del dact
            w = self.generator1[ii].weight
            b = self.generator1[ii].bias
            y = x @ w.T + b
            x = y
            w_list1.append(w.T)
            del y, w, b
            x = x + x_tmp
            dact = self.dact(x)
            dact_list1.append(dact.transpose(0, 1))
            x  = self.act(x)
            del dact
        w = self.generator[-2].weight
        b = self.generator[-2].bias
        y = x @ w.T + b
        x = y
        w_list1.append(w.T)
        del y, w, b
        dact = self.dact(x)
        dact_list1.append(dact.transpose(0, 1))
        x  = self.act(x)
        del dact
        w = self.generator[-1].weight
        b = self.generator[-1].bias
        y = x @ w.T + b
        x = y
        w_list1.append(w.T)
        del y, w, b
        actout  = self.actout(x)
        dactout = 0.1 * actout * (1 - actout)
        dact_list1.append(dactout.transpose(0, 1))
        x = actout
        del actout, dactout
        dact_list1.reverse()
        w_list1.reverse()
        dx = w_list1[0] * dact_list1[0]
        dx = dact_list1[1] * dx
        dx = w_list1[1] @ dx
        for ii in range(self.nl2):
            dx =  dact_list1[2*ii+2]*dx
            dx_tmp =  w_list1[2*ii+2]@dx
            dx = w_list1[2 * ii + 3] @ (dact_list1[2 * ii + 3] * dx_tmp) + dx
        dx0 = dx[:128,:]
        dx1 = dx[128:, :]
        dx00 = s0*dx0
        dx01 = s1*dx1
        dx10 = s1*dx0
        dx11 = s0*dx1
        dx = torch.cat((dx00 + dx01, dx10 + dx11), dim=1)
        dact_list0.reverse()
        w_list0.reverse()
        dx = w_list0[0] @ dx
        for ii in range(0,self.nl1-1):
            dx = dact_list0[2 * ii] * dx
            dx_tmp = w_list0[2 * ii + 1] @ dx
            dx = w_list0[2 * ii + 2] @ (dact_list0[2 * ii + 1] * dx_tmp) + dx
        dx = w_list0[2 * self.nl1 - 1] @ (dact_list0[2 * (self.nl1 - 1)] * dx)
        dx = dact_list0[2 * self.nl1 - 1] * dx
        dx = w_list0[2 * self.nl1] @ dx
        dx0 = dx[:, :size]
        dx1 = dx[:, size:]
        dx = torch.cat((dx0, dx1), dim=0).transpose(0, 1)
        return x, dx, coords

    def init_laplace(self, x, w, b):
        y = x @ w.T + b
        x = self.act(y)
        dact = self.dact(y)
        ddact = 10 * dact * (1 - dact)
        dx = w.T * dact
        lx = w.T * w.T * ddact
        del y, w, b, dact, ddact
        return x, dx, lx

    def linear_laplace(self, x, dx, lx, w, b):
        y = x @ w.T + b
        x = y
        dxw = dx @ w.T
        dx = dxw
        lx_b = lx @ w.T
        lx = lx_b
        del y, w, b, dxw, lx_b
        return x, dx, lx

    def act_laplace(self, x, dx, lx):
        dact = self.dact(x)
        ddact = 10 * dact * (1 - dact)
        lx_a = (dx * dx) * ddact
        lx_b = lx * dact
        lx = lx_a + lx_b
        dx = dx * dact
        x = self.act(x)
        del lx_a, lx_b, dact, ddact
        return x, dx, lx

    def actout_laplace(self, x, dx, lx):
        actout = self.actout(x)
        dactout = 0.1 * actout * (1 - actout)
        ddactout = 0.1 * dactout * (1 - 2 * actout)
        lx_a = dx * dx * ddactout
        lx_b = lx * dactout
        lx = lx_a + lx_b
        dx = dx * dactout
        x = actout
        del actout, lx_a, lx_b, dactout, ddactout
        return x, dx, lx

    def out_laplace(self, coords, B):
        x0 = coords[:, :, : self.dim]
        x1 = coords[:, :, self.dim :]
        x_size = x0.shape[1]
        x = torch.cat((x0, x1), dim=1)
        x = x.unsqueeze(2)
        x, dx, lx = self.input_mapping_laplace(x, B)
        w = self.encoder[0].weight
        b = self.encoder[0].bias
        x, dx, lx = self.linear_laplace(x, dx, lx, w, b)
        x, dx, lx = self.act_laplace(x, dx, lx)
        for ii in range(1, self.nl1):
            x_tmp, dx_tmp, lx_tmp = x, dx, lx
            w = self.encoder[ii].weight
            b = self.encoder[ii].bias
            x, dx, lx = self.linear_laplace(x, dx, lx, w, b)
            x, dx, lx = self.act_laplace(x, dx, lx)
            w = self.encoder1[ii].weight
            b = self.encoder1[ii].bias
            x, dx, lx = self.linear_laplace(x, dx, lx, w, b)
            x, dx, lx = x + x_tmp, dx + dx_tmp, lx + lx_tmp
            x, dx, lx = self.act_laplace(x, dx, lx)
        w = self.encoder[-1].weight
        b = self.encoder[-1].bias
        x, dx, lx = self.linear_laplace(x, dx, lx, w, b)
        x0 = x[:, :x_size, ...]
        x1 = x[:, x_size:, ...]
        dx0 = dx[:, :x_size, ...]
        dx1 = dx[:, x_size:, ...]
        lx0 = lx[:, :x_size, ...]
        lx1 = lx[:, x_size:, ...]
        xx = torch.cat((x0, x1), dim=2)
        x_0 = torch.logsumexp(self.scale * xx, 2) / self.scale
        x_1 = -torch.logsumexp(-self.scale * xx, 2) / self.scale
        x = torch.cat((x_0, x_1), 2).unsqueeze(2)
        x0_1 = x0 - x1
        del x0, x1
        s0 = self.dact(x0_1)
        s1 = 1.0 - s0
        dx00 = dx0 * s0
        dx01 = dx1 * s1
        dx10 = dx0 * s1
        dx11 = dx1 * s0
        dx_0 = torch.cat((dx00, dx01), dim=2)
        dx_1 = torch.cat((dx10, dx11), dim=2)
        del dx00, dx01, dx10, dx11
        dx = torch.cat((dx_0, dx_1), dim=3)
        del dx_0, dx_1
        s = 10 * s0 * s1
        lx00 = ((dx0) * s) * dx0
        lx11 = ((dx1) * s) * dx1
        lx_00_0 = lx00 + lx0 * s0
        lx_11_0 = lx11 + lx1 * s1
        lx_00_1 = -lx00 + lx0 * s1
        lx_11_1 = -lx11 + lx1 * s0
        del dx0, dx1, lx0, lx1
        del lx00, lx11, s0, s1
        lx_0 = torch.cat((lx_00_0, lx_11_0), dim=2)
        lx_1 = torch.cat((lx_00_1, lx_11_1), dim=2)
        del lx_00_0, lx_11_0, lx_00_1, lx_11_1
        lx = torch.cat((lx_0, lx_1), dim=3)
        del lx_0, lx_1
        for ii in range(self.nl2):
            x_tmp, dx_tmp, lx_tmp = x, dx, lx
            w = self.generator[ii].weight
            b = self.generator[ii].bias
            x, dx, lx = self.linear_laplace(x, dx, lx, w, b)
            x, dx, lx = self.act_laplace(x, dx, lx)
            w = self.generator1[ii].weight
            b = self.generator1[ii].bias
            x, dx, lx = self.linear_laplace(x, dx, lx, w, b)
            x, dx, lx = x+x_tmp, dx+dx_tmp, lx+lx_tmp
            x, dx, lx = self.act_laplace(x, dx, lx)
        w = self.generator[-2].weight
        b = self.generator[-2].bias
        x, dx, lx = self.linear_laplace(x, dx, lx, w, b)
        x, dx, lx = self.act_laplace(x, dx, lx)
        w = self.generator[-1].weight
        b = self.generator[-1].bias
        x, dx, lx = self.linear_laplace(x, dx, lx, w, b)
        x, dx, lx = self.actout_laplace(x, dx, lx)
        x = x.squeeze(3)
        dx = dx.squeeze(3)
        lx = lx.squeeze(3)
        return x, dx, lx, coords

    def forward(self, coords, B):
        coords = coords.clone().detach().requires_grad_(True)
        output, coords = self.out(coords, B)
        return output, coords


class Model():
    def __init__(self, ModelPath, DataPath, dim, length, device='cpu'):

        # ======================= JSON Template =======================
        self.Params = {}
        self.Params['ModelPath'] = ModelPath
        self.Params['DataPath'] = DataPath
        self.dim = dim
        self.len = length

        # Pass the JSON information
        self.Params['Device'] = device
        self.Params['Pytorch Amp (bool)'] = False

        self.Params['Network'] = {}
        self.Params['Network']['Normlisation'] = 'OffsetMinMax'

        self.Params['Training'] = {}
        self.Params['Training']['Number of sample points'] = 2e5
        self.Params['Training']['Batch Size'] = 2
        self.Params['Training']['Validation Percentage'] = 10
        self.Params['Training']['Number of Epochs'] = 10000
        self.Params['Training']['Resampling Bounds'] = [0.1, 0.9]
        self.Params['Training']['Print Every * Epoch'] = 1
        self.Params['Training']['Save Every * Epoch'] = 100
        self.Params["Training"]["Learning Rate"] = 1e-3
        self.Params['Training']['Random Distance Sampling'] = True
        self.Params['Training']['Use Scheduler (bool)'] = False

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss = []

    def gradient(self, y, x, create_graph=True):

        grad_y = torch.ones_like(y)
        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        return grad_x

    def Loss(self, points, Yobs, B, beta, gamma):
        print("Calculating Loss")
        tau, dtau, ltau, Xp = self.network.out_laplace(points, B)
        D = Xp[:,:,self.dim:]-Xp[:,:,:self.dim]
        T0 = torch.einsum("bij,bij->bi", D, D)
        lap0 = ltau[:,:,:self.dim].sum(-1)
        lap1 = ltau[:, :, self.dim :].sum(-1)
        DT0 = dtau[:,:,:self.dim]
        DT1 = dtau[:, :, self.dim :]
        T01 = T0*torch.einsum('bij,bij->bi', DT0, DT0)
        T02 = -2 * tau[:, :, 0] * torch.einsum("bij,bij->bi", DT0, D)
        T11 = T0*torch.einsum('bij,bij->bi', DT1, DT1)
        T12 = 2 * tau[:, :, 0] * torch.einsum("bij,bij->bi", DT1, D)
        T3 = tau[:, :, 0] ** 2
        S0 = (T01-T02+T3)
        S1 = T11 - T12 + T3
        sq_Ypred0 = 1 / (torch.sqrt(S0) / T3 + gamma * lap0)
        sq_Ypred1 = 1 / (torch.sqrt(S1) / T3 + gamma * lap1)
        sq_Yobs0 = Yobs[:, :, 0]
        sq_Yobs1 = Yobs[:, :, 1]
        loss0 = sq_Ypred0 / sq_Yobs0 + sq_Yobs0 / sq_Ypred0
        loss1 = sq_Ypred1 / sq_Yobs1 + sq_Yobs1 / sq_Ypred1
        diff = loss0 + loss1 - 4
        loss_n = (
            torch.sum((loss0 + loss1 - 4)) / Yobs.shape[0] / Yobs.shape[1]
            + 0.01 * torch.norm(B) ** 2 / Yobs.shape[0] / Yobs.shape[1]
        )
        loss = beta * loss_n
        return loss, loss_n, diff

    def train(self):
        self.network = NN(self.Params['Device'],self.dim)
        self.network.apply(self.network.init_weights)
        self.network.to(self.Params["Device"])
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.Params['Training']['Learning Rate']
            ,weight_decay=0.1)
        if self.Params['Training']['Use Scheduler (bool)'] == True:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.dataset = db.Database(
            self.Params["DataPath"], torch.device(self.Params["Device"]), self.len
        )
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.Params["Training"]["Batch Size"]),
            num_workers=1,
            shuffle=True,
        )
        beta = 1.0
        prev_diff = 1.0
        current_diff = 1.0
        step = -2000.0 / 4000.0
        tt = time.time()
        current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
        current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
        prev_state_queue = []
        prev_optimizer_queue = []
        inner_batch = 10000
        inner_rows = self.dataset[0][0].shape[0]
        inner_size = int(inner_rows / inner_batch)

        for epoch in range(1, self.Params["Training"]["Number of Epochs"] + 1):
            total_train_loss = 0
            total_diff = 0
            alpha = min(max(0.5, 0.5 + 0.5 * step), 1.07)
            step += 1.0 / 4000 / ((int)(epoch / 4000) + 1.0)
            gamma = 0.001
            prev_state_queue.append(current_state)
            prev_optimizer_queue.append(current_optimizer)
            if(len(prev_state_queue)>5):
                prev_state_queue.pop(0)
                prev_optimizer_queue.pop(0)
            current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
            current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
            self.optimizer.param_groups[0]["lr"] = np.clip(
                1e-3 * (1 - (epoch - 8000) / 1000.0), a_min=5e-4, a_max=1e-3
            )
            prev_diff = current_diff
            iter = 0
            while True:
                total_train_loss = 0
                total_diff = 0
                for _, data_tuple in enumerate(dataloader, 0):
                    data = data_tuple[0].to(self.Params["Device"])
                    B = data_tuple[1].to(self.Params["Device"])
                    points = data[:, :, : 2 * self.dim]
                    speed = data[:, :, 2 * self.dim :]
                    speed = alpha * speed + 1 - alpha
                    idx0 = torch.randperm(inner_rows)
                    idx1 = torch.randperm(inner_rows)
                    points_shuffled = torch.cat((points[0,idx0,:].unsqueeze(0), points[1,idx1,:].unsqueeze(0)), dim=0)
                    speed_shuffled = torch.cat(
                        (
                            speed[0, idx0, :].unsqueeze(0),
                            speed[1, idx1, :].unsqueeze(0),
                        ),
                        dim=0,
                    )
                    for ii in range(inner_size):
                        if ii > 5:
                            break
                        batch_points =  points_shuffled[:,ii*inner_batch:(ii+1)*inner_batch,:]
                        batch_speed = speed_shuffled[
                            :, ii * inner_batch : (ii + 1) * inner_batch, :
                        ]
                        batch_points.requires_grad_()
                        self.B = B[0,:]
                        loss_value, loss_n, wv = self.Loss(
                            batch_points, batch_speed, B, beta, gamma
                        )
                        loss_value.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        total_train_loss += loss_value.clone().detach()
                        total_diff += loss_n.clone().detach()
                        del batch_points, batch_speed, loss_value, loss_n
                    del points, speed, points_shuffled, speed_shuffled

                total_train_loss /= len(dataloader) * 5.0
                total_diff /= len(dataloader) * 5.0
                current_diff = total_diff
                diff_ratio = current_diff / prev_diff
                if diff_ratio < 1.2 and diff_ratio > 0:
                    # This is eta in the P-NTFields paper. The Loss should not increase too much to be taken into account.
                    # If the loss has increased by more than 20%, the gradient is discarded and a new dropout is randomized.
                    break
                else:
                    iter += 1
                    with torch.no_grad():
                        random_number = random.randint(0, 4)
                        self.network.load_state_dict(
                            prev_state_queue[random_number], strict=True
                        )
                        self.optimizer.load_state_dict(
                            prev_optimizer_queue[random_number]
                        )
                    print(
                        "RepeatEpoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                            epoch, total_diff, alpha
                        )
                    )

            self.total_train_loss.append(total_train_loss)
            beta = 1.0 / total_diff
            if self.Params['Training']['Use Scheduler (bool)'] == True:
                self.scheduler.step(total_train_loss)
            if epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                with torch.no_grad():
                    # print("Epoch = {} -- Training loss = {:.4e} -- Validation loss = {:.4e}".format(
                    #    epoch, total_train_loss, total_val_loss))
                    print("Epoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                        epoch, total_diff.item(), alpha))

            if (epoch % self.Params['Training']['Save Every * Epoch'] == 0) or (epoch == self.Params['Training']['Number of Epochs']) or (epoch == 1):
                self.plot(epoch,total_diff.item(),alpha)
                with torch.no_grad():
                    self.save(epoch=epoch, val_loss=total_diff)

    def save(self, epoch='', val_loss=''):
        """Saving an instance of the model"""
        torch.save({'epoch': epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'B_state_dict':self.B,
                    'train_loss': self.total_train_loss,
                    'val_loss': self.total_val_loss}, '{}/Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(self.Params['ModelPath'], str(epoch).zfill(5), val_loss))

    def load(self, filepath):
        checkpoint = torch.load(
            filepath, map_location=torch.device(self.Params["Device"])
        )
        self.network = NN(self.Params['Device'],self.dim)
        self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.network.to(torch.device(self.Params['Device']))
        self.network.float()
        self.network.eval()

    def load_pretrained_state_dict(self, state_dict):
        own_state = self.state_dict

    def TravelTimes(self, Xp):
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.Params['Device']))
        tau, _ = self.network.out(Xp, self.B)
        D = Xp[:, self.dim :] - Xp[:, : self.dim]
        T0 = torch.einsum("ij,ij->i", D, D)
        TT = torch.sqrt(T0) / tau[:, 0]
        del Xp, tau, T0
        return TT

    def Tau(self, Xp):
        Xp = Xp.to(torch.device(self.Params["Device"]))
        tau, _ = self.network.out(Xp, self.B)
        return tau

    def Speed(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
        tau, dtau, _ = self.network.out_grad(Xp, self.B)
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        T0 = torch.einsum("ij,ij->i", D, D)
        DT1 = dtau[:, self.dim :]
        T1 = T0*torch.einsum('ij,ij->i', DT1, DT1)
        T2 = 2*tau[:,0]*torch.einsum('ij,ij->i', DT1, D)
        T3 = tau[:, 0] ** 2
        S = T1 - T2 + T3
        Ypred = T3 / torch.sqrt(S)
        del Xp, tau, dtau, T0, T1, T2, T3
        return Ypred

    def Gradient(self, Xp, B):
        Xp = Xp.to(torch.device(self.Params['Device']))
        tau, dtau, _ = self.network.out_backgrad(Xp, B)
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D))
        T3 = tau[:, 0] ** 2
        V0 = D
        V1 = dtau[:, self.dim :]
        Y1 = 1 / (T0 * tau[:, 0]).unsqueeze(1) * V0
        Y2 = (T0 / T3).unsqueeze(1) * V1
        Ypred1 = -(Y1-Y2)
        Spred1 = torch.norm(Ypred1, p=2, dim =1).unsqueeze(1)
        Ypred1 = 1 / Spred1**2 * Ypred1
        V0=-D
        V1 = dtau[:, : self.dim]
        Y1 = 1 / (T0 * tau[:, 0]).unsqueeze(1) * V0
        Y2 = (T0 / T3).unsqueeze(1) * V1
        Ypred0 = -(Y1-Y2)
        Spred0 = torch.norm(Ypred0, p=2, dim=1).unsqueeze(1)
        Ypred0 = 1/Spred0**2*Ypred0

        return torch.cat((Ypred0, Ypred1),dim=1)

    def plot(self, epoch, total_train_loss, alpha):
        limit = 0.5
        xmin = [-limit, -limit]
        xmax = [limit, limit]
        spacing = limit / 40.0
        X, Y = np.meshgrid(
            np.arange(xmin[0], xmax[0], spacing), np.arange(xmin[1], xmax[1], spacing)
        )
        Xsrc = [0] * self.dim
        Xsrc[0] = -0.25
        Xsrc[1] = -0.25
        XP = np.zeros((len(X.flatten()), 2 * self.dim))
        XP[:, : self.dim] = Xsrc
        XP[:, self.dim + 0] = X.flatten()
        XP[:, self.dim + 1] = Y.flatten()
        XP = Variable(Tensor(XP)).to(self.Params["Device"])
        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)#*5
        tau = self.Tau(XP)
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        V  = ss.to('cpu').data.numpy().reshape(X.shape)
        TAU = tau.to("cpu").data.numpy().reshape(X.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X, Y, V, vmin=0, vmax=1)
        ax.contour(X, Y, TT, np.arange(0, 3, 0.05), cmap="bone", linewidths=0.5)
        plt.colorbar(quad1, ax=ax, pad=0.1, label="Predicted Velocity")
        plt.savefig(
            self.Params["ModelPath"]
            + "/plots"
            + str(epoch)
            + "_"
            + str(alpha)
            + "_"
            + str(round(total_train_loss, 4))
            + "_0.jpg",
            bbox_inches="tight",
        )
        plt.close(fig)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X, Y, TAU, vmin=0, vmax=1)
        ax.contour(X, Y, TT, np.arange(0, 3, 0.05), cmap="bone", linewidths=0.5)
        plt.colorbar(quad1, ax=ax, pad=0.1, label="Predicted Velocity")
        plt.savefig(
            self.Params["ModelPath"]
            + "/tauplots"
            + str(epoch)
            + "_"
            + str(alpha)
            + "_"
            + str(round(total_train_loss, 4))
            + "_0.jpg",
            bbox_inches="tight",
        )
        plt.close(fig)
