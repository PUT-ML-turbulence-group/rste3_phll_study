import itertools
from typing import Generator

import torch
from e3nn import o3
from e3nn.nn import NormActivation
from torch.nn import Module, ModuleList

from lightning.model_backbones.e3_layers import SplitModule, TPWrap, ACT_DICT, InteractionBlock, ScalarReadoutBlock


class DeepRSTPredictorAct(Module):
    def __init__(self, irreps_in, irreps_out, **kwargs):
        super(DeepRSTPredictorAct, self).__init__()
        print('Predictor with additional activations')
        self.model_args = ['x']
        hidden_irreps = kwargs['hidden_irreps']
        if type(hidden_irreps) == str:
            hidden_irreps = [hidden_irreps]
        if len(hidden_irreps) == 1:
            hidden_irreps = hidden_irreps + hidden_irreps

        self.irreps_out = irreps_out
        self.act = NormActivation(hidden_irreps[0], torch.nn.functional.silu)
        self.lin1 = o3.Linear(irreps_in, hidden_irreps[1], biases=True)
        self.lin2 = o3.Linear(hidden_irreps[1], hidden_irreps[0], biases=True)  # interaction 1
        self.split1 = SplitModule(hidden_irreps[0], irreps_out=hidden_irreps[0], biases=True)  # interaction 1
        self.tp1 = o3.TensorProduct(hidden_irreps[0], hidden_irreps[0], hidden_irreps[0],  # interaction 1
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

        self.lin3 = o3.Linear(hidden_irreps[0], hidden_irreps[0], biases=True)  # interaction 2
        self.split2 = SplitModule(hidden_irreps[0], biases=True)  # interaction 2
        self.tp2 = o3.TensorProduct(hidden_irreps[0], hidden_irreps[0], hidden_irreps[0],  # interaction 2
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

        self.lin4 = o3.Linear(hidden_irreps[0], hidden_irreps[0], biases=True)  # interaction 3
        self.split3 = SplitModule(hidden_irreps[0], biases=True)  # interaction 3
        self.tp3 = o3.TensorProduct(hidden_irreps[0], hidden_irreps[0], hidden_irreps[0],  # interaction 3
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

        self.lin5 = o3.Linear(hidden_irreps[0], hidden_irreps[0], biases=True)
        self.split4 = SplitModule(hidden_irreps[0], biases=True)
        self.tp4 = o3.FullyConnectedTensorProduct(hidden_irreps[0], hidden_irreps[0], irreps_out)

    def forward(self, data):
        h = self.lin1(data)
        h = self.act(self.lin2(h))
        ha, hb = self.split1(h)
        ht = self.tp1(ha, hb)

        h = self.act(self.lin3(ht))
        ha, hb = self.split2(h)
        ht = self.tp2(ha, hb)

        h = self.act(self.lin4(ht))
        ha, hb = self.split3(h)
        ht = self.tp3(ha, hb)

        h = self.lin5(ht)
        ha, hb = self.split4(h)
        ht = self.tp4(ha, hb)

        return ht


class TurbulenceNN(Module):
    def __init__(self, irreps_in, irreps_out, hidden_irreps, activation_fn='silu', split_activation_fn='sigmoid'):
        super(TurbulenceNN, self).__init__()
        print('Predictor with additional activations')
        self.model_args = ['x']
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.hidden_irreps = o3.Irreps(irreps_out)

        #activation_fn = ACT_DICT[activation_fn]
        #split_activation_fn = ACT_DICT[split_activation_fn]
        self.activation_fn = activation_fn
        self.split_activation_fn = split_activation_fn

        self.act = NormActivation(hidden_irreps, activation_fn)
        self.lin1 = o3.Linear(irreps_in, hidden_irreps, biases=True)
        self.block1 = InteractionBlock(hidden_irreps, path_type='uvu', trainable=True, activation_fn=activation_fn,
                                       split_activation_fn=split_activation_fn)
        self.block2 = InteractionBlock(hidden_irreps, path_type='uvu', trainable=True, activation_fn=activation_fn,
                                       split_activation_fn=split_activation_fn)
        self.block3 = InteractionBlock(hidden_irreps, path_type='uvu', trainable=True, activation_fn=activation_fn,
                                       split_activation_fn=split_activation_fn)

        self.lin5 = o3.Linear(hidden_irreps, hidden_irreps, biases=True)
        self.split4 = SplitModule(hidden_irreps, biases=True)
        self.tp4 = o3.FullyConnectedTensorProduct(hidden_irreps, hidden_irreps, irreps_out)

    def forward(self, data):
        h = self.lin1(data)

        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)

        h = self.lin5(h)
        ha, hb = self.split4(h)
        ht = self.tp4(ha, hb)

        return ht


class ScalarTurbulenceNN(Module):
    def __init__(self, irreps_in, irreps_out, hidden_irreps, activation_fn='silu', split_activation_fn='sigmoid'):
        super(ScalarTurbulenceNN, self).__init__()
        print('Predictor with additional activations')
        self.model_args = ['x']
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.hidden_irreps = o3.Irreps(irreps_out)

        #activation_fn = ACT_DICT[activation_fn]
        #split_activation_fn = ACT_DICT[split_activation_fn]
        self.activation_fn = activation_fn
        self.split_activation_fn = split_activation_fn

        self.act = NormActivation(hidden_irreps, activation_fn)
        self.lin1 = o3.Linear(irreps_in, hidden_irreps, biases=True)
        self.block1 = InteractionBlock(hidden_irreps, path_type='uvu', trainable=True, activation_fn=activation_fn,
                                       split_activation_fn=split_activation_fn)
        self.block2 = InteractionBlock(hidden_irreps, path_type='uvu', trainable=True, activation_fn=activation_fn,
                                       split_activation_fn=split_activation_fn)
        self.readout = ScalarReadoutBlock(hidden_irreps, '64x0e', '1x0e')
        self.lin2 = o3.Linear(hidden_irreps, hidden_irreps, biases=True)


    def forward(self, data):
        h = self.lin1(data)
        #h = self.act(h)

        h = self.block1(h)
        h = self.block2(h)

        h = self.lin2(h)

        out = self.readout(h)

        return out



if __name__ == '__main__':
    block = InteractionBlock('32x0e + 32x1e + 32x1o + 32x2e')

