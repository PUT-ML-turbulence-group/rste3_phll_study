import numpy as np
import torch

import e3nn
from e3nn import o3
from torch.nn import Module, ModuleList
from e3nn.nn import NormActivation, Gate
from e3nn.o3 import Linear


import itertools
from typing import Generator


ACT_DICT = {
    'sigmoid': torch.nn.functional.sigmoid,
    'silu': torch.nn.functional.silu

}


class SplitModule(Module):
    def __init__(self, irreps_in, irreps_out=None, irreps_mid=None, biases=True, activation_fn='sigmoid'):
        super().__init__()
        if irreps_out is None:
            irreps_out = irreps_in
        if irreps_mid is None:
            irreps_mid = irreps_out

        self.activation_fn = ACT_DICT[activation_fn]

        self.lin1 = o3.Linear(irreps_in, irreps_mid, biases=biases)
        self.lin1a = o3.Linear(irreps_in, irreps_mid, biases=biases)
        self.bn1a = e3nn.nn.BatchNorm(irreps_mid)
        self.act1a = NormActivation(irreps_mid, self.activation_fn)


    def forward(self, x):
        h = self.lin1(x)
        ha = self.act1a(self.bn1a(self.lin1a(x)))
        return h, ha


class TPWrap(Module):
    def __init__(self, irreps_in_1, irreps_in_2=None, irreps_out=None, path_type='uvu', trainable=True):
        super(TPWrap, self).__init__()
        if irreps_in_2 is None:
            irreps_in_2 = irreps_in_1
        if irreps_in_1 is None:
            irreps_in_1 = irreps_in_2

        self.irreps_in_1 = o3.Irreps(irreps_in_1)
        self.irreps_in_2 = o3.Irreps(irreps_in_2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.path_type = path_type
        self.trainable = trainable

        self.paths = self.create_paths()
        self.tp = o3.TensorProduct(irreps_in_1, irreps_in_2, irreps_out, self.paths)

    def create_paths(self):
        irrep_types_1 = [irr.ir for irr in self.irreps_in_1]
        irrep_types_2 = [irr.ir for irr in self.irreps_in_2]
        irrep_types_out = [irr.ir for irr in self.irreps_out]

        irrep_combinations: Generator = itertools.product(range(len(irrep_types_1)), range(len(irrep_types_2)))

        paths = []
        for ind1, ind2 in irrep_combinations:
            irrep_products = irrep_types_1[ind1] * irrep_types_2[ind2]
            #irrep_products = list(irrep_products)
            #print(irrep_products, irrep_types_1[ind1], irrep_types_2[ind2])
            for prod in irrep_products:
                if prod in irrep_types_out:
                    match = irrep_types_out.index(prod)
                    path = (ind1, ind2, match, self.path_type, self.trainable)
                    paths.append(path)
        return paths

    def forward(self, x1, x2):
        return self.tp(x1, x2)


class InteractionBlock(Module):
    def __init__(self, irreps, path_type='uvu', trainable=True, activation_fn='silu', split_activation_fn='sigmoid'):
        super(InteractionBlock, self).__init__()
        self.irreps = o3.Irreps(irreps)
        self.irrep_types = [irr.ir for irr in self.irreps]
        self.path_type = path_type
        self.trainable = trainable
        self.paths = self.create_paths()

        self.linear = o3.Linear(irreps, irreps)
        self.act = ACT_DICT[activation_fn]
        self.split = SplitModule(irreps,
                                 activation_fn=split_activation_fn)
        self.tp = o3.TensorProduct(irreps, irreps, irreps, self.paths)

    def create_paths(self):
        irrep_types = self.irrep_types
        irrep_combinations: Generator = itertools.product(range(len(irrep_types)), range(len(irrep_types)))

        paths = []
        for ind1, ind2 in irrep_combinations:
            irrep_products = irrep_types[ind1] * irrep_types[ind2]
            for prod in irrep_products:
                if prod in irrep_types:
                    match = irrep_types.index(prod)
                    paths.append((ind1, ind2, match, self.path_type, self.trainable))
        return paths

    def forward(self, x):
        proj = self.linear(x)
        proj_a = self.act(proj)
        h, ha = self.split(proj_a)
        out = self.tp(h, ha)
        return out


class ScalarReadoutBlock(Module):
    def __init__(self, irreps_in, irreps_produced, irreps_out, num_linear=3):
        super(ScalarReadoutBlock, self).__init__()
        self.tp = TPWrap(irreps_in, irreps_in, irreps_produced, path_type='uvw')
        self.linear_layers = ModuleList()
        for _ in range(num_linear - 1):
            self.linear_layers.append(o3.Linear(irreps_produced, irreps_produced))
        self.linear_layers.append(o3.Linear(irreps_produced, irreps_out))

    def forward(self, x):
        h = self.tp(x, x)
        for i in range(len(self.linear_layers)):
            h = self.linear_layers[i](h)
        return h


if __name__ == '__main__':
    tp = TPWrap('16x0e + 16x1e', '8x0e + 8x0o + 8x1o', '16x0e + 16x1e')
