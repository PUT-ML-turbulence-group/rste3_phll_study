import torch
from torch.nn import Module, ModuleList
import e3nn
from e3nn import o3
from e3nn.nn import Activation, NormActivation

from model.layers import SplitModule, LayerE3, HalfSplit, NormActivationWrapper
from utils import gating_representation, scalar_counter, full_to_diagonal, triangle_to_full, unit_function

class DeepRSTPredictor(Module):
    def __init__(self, irreps_in, irreps_out, **kwargs):
        super(DeepRSTPredictor, self).__init__()
        # hidden_irreps = ['5x0e + 2x1e + 3x1o + 3x2e']

        self.model_args = ['x']

        hidden_irreps = kwargs['hidden_irreps']
        if type(hidden_irreps) == str:
            hidden_irreps = [hidden_irreps]
        if len(hidden_irreps) == 1:
            hidden_irreps = hidden_irreps + hidden_irreps

        self.irreps_out = irreps_out
        self.lin1 = o3.Linear(irreps_in, hidden_irreps[1], biases=True)
        self.lin2 = o3.Linear(hidden_irreps[1], hidden_irreps[0], biases=True)
        self.split1 = SplitModule(hidden_irreps[0], irreps_out=hidden_irreps[0], biases=True)
        self.tp1 = o3.TensorProduct(hidden_irreps[0], hidden_irreps[0], hidden_irreps[0],
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

        self.lin3 = o3.Linear(hidden_irreps[0], hidden_irreps[0], biases=True)
        self.split2 = SplitModule(hidden_irreps[0], biases=True)
        self.tp2 = o3.TensorProduct(hidden_irreps[0], hidden_irreps[0], hidden_irreps[0],
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

        self.lin4 = o3.Linear(hidden_irreps[0], hidden_irreps[0], biases=True)
        self.split3 = SplitModule(hidden_irreps[0], biases=True)
        self.tp3 = o3.TensorProduct(hidden_irreps[0], hidden_irreps[0], hidden_irreps[0],
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
        h = self.lin2(h)
        ha, hb = self.split1(h)
        ht = self.tp1(ha, hb)

        h = self.lin3(ht)
        ha, hb = self.split2(h)
        ht = self.tp2(ha, hb)

        h = self.lin4(ht)
        ha, hb = self.split3(h)
        ht = self.tp3(ha, hb)

        h = self.lin5(ht)
        ha, hb = self.split4(h)
        ht = self.tp4(ha, hb)

        return ht



class DeepRSTPredictorLin(Module):
    def __init__(self, irreps_in, irreps_out, **kwargs):
        super(DeepRSTPredictorLin, self).__init__()
        # hidden_irreps = ['5x0e + 2x1e + 3x1o + 3x2e']

        self.model_args = ['x']

        hidden_irreps = kwargs['hidden_irreps']
        if type(hidden_irreps) == str:
            hidden_irreps = [hidden_irreps]
        if len(hidden_irreps) == 1:
            hidden_irreps = hidden_irreps + hidden_irreps

        self.irreps_out = irreps_out
        self.lin1 = o3.Linear(irreps_in, hidden_irreps[1], biases=True)
        self.lin2 = o3.Linear(hidden_irreps[1], hidden_irreps[0], biases=True)
        self.split1 = SplitModule(hidden_irreps[0], irreps_out=hidden_irreps[0], biases=True)
        self.tp1 = o3.TensorProduct(hidden_irreps[0], hidden_irreps[0], hidden_irreps[0],
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

        self.lin3 = o3.Linear(hidden_irreps[0], hidden_irreps[0], biases=True)
        self.split2 = SplitModule(hidden_irreps[0], biases=True)
        self.tp2 = o3.TensorProduct(hidden_irreps[0], hidden_irreps[0], hidden_irreps[0],
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

        self.lin4 = o3.Linear(hidden_irreps[0], hidden_irreps[0], biases=True)
        self.split3 = SplitModule(hidden_irreps[0], biases=True)
        self.tp3 = o3.TensorProduct(hidden_irreps[0], hidden_irreps[0], hidden_irreps[0],
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
        self.lin6 = o3.Linear(hidden_irreps[0], hidden_irreps[0], biases=True)
        self.lin7 = o3.Linear(hidden_irreps[0], hidden_irreps[0], biases=True)



    def forward(self, data):
        h = self.lin1(data)
        h = self.lin2(h)
        ha, hb = self.split1(h)
        ht = self.tp1(ha, hb)

        h = self.lin3(ht)
        ha, hb = self.split2(h)
        ht = self.tp2(ha, hb)

        h = self.lin4(ht)
        ha, hb = self.split3(h)
        ht = self.tp3(ha, hb)

        h = self.lin5(ht)
        h1 = self.lin6(h)
        h2 = self.lin7(h1)

        return h2


class DeepRSTPredictorAct(Module):
    def __init__(self, irreps_in, irreps_out, **kwargs):
        super(DeepRSTPredictorAct, self).__init__()
        # hidden_irreps = ['5x0e + 2x1e + 3x1o + 3x2e']
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
        self.lin2 = o3.Linear(hidden_irreps[1], hidden_irreps[0], biases=True)
        self.split1 = SplitModule(hidden_irreps[0], irreps_out=hidden_irreps[0], biases=True)
        self.tp1 = o3.TensorProduct(hidden_irreps[0], hidden_irreps[0], hidden_irreps[0],
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

        self.lin3 = o3.Linear(hidden_irreps[0], hidden_irreps[0], biases=True)
        self.split2 = SplitModule(hidden_irreps[0], biases=True)
        self.tp2 = o3.TensorProduct(hidden_irreps[0], hidden_irreps[0], hidden_irreps[0],
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

        self.lin4 = o3.Linear(hidden_irreps[0], hidden_irreps[0], biases=True)
        self.split3 = SplitModule(hidden_irreps[0], biases=True)
        self.tp3 = o3.TensorProduct(hidden_irreps[0], hidden_irreps[0], hidden_irreps[0],
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

