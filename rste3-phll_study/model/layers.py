import numpy as np
import torch

import e3nn
from e3nn import o3
from torch.nn import Module, ModuleList
from e3nn.nn import NormActivation, Gate
from e3nn.o3 import Linear

from utils import gating_representation, gate_irreps

class LayerE3(Module):
    def __init__(self, irreps_in, irreps_out=None, sequence='', tensor_map='uvu', activation_fn=''):
        super(LayerE3, self).__init__()
        if type(irreps_in) == str:
            irreps_in = e3nn.o3.Irreps(irreps_in)
        if irreps_out is None:
            irreps_out = irreps_in
        self.layer_content = torch.nn.ModuleList()
        for i, c in enumerate(sequence):
            double_the_layer = False
            if i + 1 < len(sequence):
                if sequence[i + 1] == 'a' and activation_fn == 'maxout':
                    double_the_layer = True
            if c == 't':
                if double_the_layer:
                    self.layer_content.append(DoubleLayerE3(TensorProductWrapper, irreps_in, irreps_out, tensor_map))
                else:
                    self.layer_content.append(TensorProductWrapper(irreps_in, irreps_out, tensor_map))
            if c == 'l':
                if double_the_layer:
                    self.layer_content.append(DoubleLayerE3(e3nn.o3.Linear, irreps_in, irreps_out))
                else:
                    self.layer_content.append(e3nn.o3.Linear(irreps_in, irreps_out))
            if c == 'a':
                if double_the_layer:
                    self.layer_contentuf.append(DoubleLayerE3(NormActivationWrapper, irreps_in, activation_fn))
                else:
                    self.layer_content.append(NormActivationWrapper(irreps_in, activation_fn))
            if c == 's':
                pass
                # self.layer_content.append(ToBeImplementedSplit(to_be_implemented_args))
            irreps_in = irreps_out

    def forward(self, data):
        for l in self.layer_content:
            data = l(data)
        return data


class TensorProductWrapper(Module):
    def __init__(self, irreps_in, irreps_out, tensor_map):
        super(TensorProductWrapper, self).__init__()
        self.l = None
        if tensor_map == 'uvw':
            self.l = e3nn.o3.FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_out)
        if tensor_map == 'uvu':
            self.l = o3.TensorProduct(irreps_in, irreps_in, irreps_out,
                                      [(0, 0, 0, 'uvu', True),
                                       (0, 1, 1, 'uvu', True),
                                       (0, 2, 2, 'uvu', True),
                                       (0, 3, 3, 'uvu', True),
                                       (1, 0, 1, 'uvu', True),
                                       (1, 1, 0, 'uvu', True),
                                       (1, 1, 1, 'uvu', True),
                                       (1, 1, 3, 'uvu', True),
                                       (2, 0, 2, 'uvu', True),
                                       (2, 1, 2, 'uvu', True),
                                       (2, 2, 0, 'uvu', True),
                                       (2, 2, 1, 'uvu', True),
                                       (2, 2, 3, 'uvu', True),
                                       (2, 3, 2, 'uvu', True)])

    def forward(self, data):
        return self.l(data, data)

def unit_function(t):
    return t/torch.where(t!=0, t, 1)


class NormActivationWrapper(Module):
    def __init__(self, irreps, activation_fn):
        super(NormActivationWrapper, self).__init__()
        self.l = None
        if activation_fn == 'maxout':
            self.l = MaxoutE3(irreps)
        else:
            self.l = e3nn.nn.NormActivation(irreps, self.get_activation_fn(activation_fn))

    def forward(self, data):
        return self.l(data)


    @staticmethod
    def get_activation_fn(activation_fn: str):
        if activation_fn == 'unit':
            return unit_function
        else:
            return getattr(torch.nn.functional, activation_fn)


class DoubleLayerE3(Module):
    def __init__(self, layer_class, *args, **kwargs):
        super(DoubleLayerE3, self).__init__()
        self.first = layer_class(*args, **kwargs)
        self.second = layer_class(*args, **kwargs)

    def forward(self, data):
        return torch.cat((self.first(data), self.second(data)), dim=-1)



class UnitLayer(torch.nn.Module):
    def __init__(self, irreps):
        super(UnitLayer, self).__init__()
        if type(irreps) == str:
            irreps = e3nn.o3.Irreps(irreps)
        self.irreps = irreps

        self.gate = e3nn.nn.Norm(irreps)

        self.gate = e3nn.nn.Gate('', [], e3nn.o3.Irreps)


class MaxoutE3(torch.nn.Module):
    def __init__(self, irreps):
        super(MaxoutE3, self).__init__()
        if type(irreps) == str:
            irreps = e3nn.o3.Irreps(irreps)
        self.irreps = irreps

        self.gate = e3nn.nn.Gate('', [], e3nn.o3.Irreps(str(irreps.num_irreps) + 'x0e'), [None],
                                 irreps)
        self.norm = e3nn.o3.Norm(irreps)

    def forward(self, data):
        # data = torch.cat([data1, data2], dim=-1).reshape(-1, 2, self.irreps.dim) cat moved to the DoubleLayerE3 class
        data = data.reshape(-1, 2, self.irreps.dim)
        norm = self.norm(data)
        weights = torch.cat([norm[:, [0], :] >= norm[:, [1], :], norm[:, [0], :] < norm[:, [1], :]], dim=1)
        gated_data = self.gate(torch.cat([weights, data], dim=-1))
        return torch.sum(gated_data.reshape(-1, 2, self.irreps.dim), dim=1)


class SplitOperation(Module):
    def __init__(self, irreps_in, split: str = 'half', op: str = 'tp', irreps_out=None, irreps_mid=None, biases=False):
        super().__init__()

        if irreps_out is None:
            irreps_out = irreps_in
        if irreps_mid is None:
            irreps_mid = irreps_out

        gate_args = gate_irreps(irreps_out)
        gate_args = [gate_args[0], [None],
                     gate_args[1], [None],
                     gate_args[2]]

        self.is_gated = False
        self.gate = None

        if split == 'full':
            self.split = SplitModule(irreps_in, irreps_out=irreps_mid, irreps_mid=irreps_mid, biases=biases)
        if split == 'half':
            self.split = HalfSplit(irreps_in, irreps_out=irreps_mid, biases=biases)
        if split == 'quarter':
            self.split = QuarterSplit(irreps_in, irreps_out=irreps_mid, biases=biases)

        if op == 'tp':
            if split == 'quarter':
                self.tp = o3.FullyConnectedTensorProduct(irreps_in, irreps_mid, irreps_out)
            else:
                self.tp = o3.FullyConnectedTensorProduct(irreps_mid, irreps_mid, irreps_out)
        if op == 'tpgate':
            self.is_gated = True
            if split == 'quarter':
                self.tp = o3.FullyConnectedTensorProduct(irreps_in, irreps_mid, gating_representation(irreps_out))
                self.gate = Gate(*gate_args)
            else:
                self.tp = o3.FullyConnectedTensorProduct(irreps_mid, irreps_mid, gating_representation(irreps_out))
                self.gate = Gate(*gate_args)

    def forward(self, data):
        h, ha = self.split(data)
        res = self.tp(h, ha)
        if self.is_gated:
            res = self.gate(res)
        return res


class SplitModule(Module):
    def __init__(self, irreps_in, irreps_out=None, irreps_mid=None, biases=True, activation_fn='sigmoid'):
        super().__init__()
        if irreps_out is None:
            irreps_out = irreps_in
        if irreps_mid is None:
            irreps_mid = irreps_out

        self.lin1 = o3.Linear(irreps_in, irreps_mid, biases=biases)
        self.lin1a = o3.Linear(irreps_in, irreps_mid, biases=biases)
        self.bn1a = e3nn.nn.BactNorm(irreps_mid)
        self.act1a = NormActivation(irreps_mid, NormActivationWrapper.get_activation_fn(activation_fn))


    def forward(self, data):
        h = self.lin1(data)
        ha = self.act1a(self.bn1a(self.lin1a(data)))
        return h, ha





class HalfSplit(Module):
    def __init__(self, irreps_in, irreps_out=None, biases=False):
        super().__init__()
        if irreps_out is None:
            irreps_out = irreps_in

        self.lin1 = o3.Linear(irreps_in, irreps_out, biases=biases)
        self.lin1a = o3.Linear(irreps_in, irreps_out, biases=biases)

        self.act1a = NormActivation(irreps_out, torch.sigmoid)

    def forward(self, data):
        h = self.lin1(data)
        ha = self.act1a(self.lin1a(data))
        return h, ha


class QuarterSplit(Module):
    def __init__(self, irreps_in, irreps_out=None, biases=False):
        super().__init__()
        if irreps_out is None:
            irreps_out = irreps_in

        self.lin1a = o3.Linear(irreps_in, irreps_out, biases=biases)

        self.act1a = NormActivation(irreps_out, torch.sigmoid)

    def forward(self, data):
        ha = self.act1a(self.lin1a(data))
        return data, ha
