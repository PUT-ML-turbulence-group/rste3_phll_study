import hydra
import numpy as np
import torch

from lightning.data_structures.data_representation import TensorsData


class EquivariantScalerWrap:
    def __init__(self, scaler_backbone, *args, **kwargs):
        """
        An equivariant scaler wrapper that makes a classic scaler act on a list of tensors inside TensorsData.
        To ensure equivariance, norms of tensors are extracted, scaled according to the embedded scaler
        and then applied back to the tensors.

        During inverse transformation, the scaler acts on a single tensor - a representation commonly yielded
        by a machine learning pipeline. In order to scale back, the wrapper has to split the data back
        into respective tensors - to achieve that, one needs to call inverse_setup and pass the list
        of irreps associated with TensorData as argument.
        TensorsData is equipped with both a list of tensors and a list of irreps. Preferably, one should use this
        view of the data to both fit the scaler and set up the inverse by provinding the associated irreps.
        """
        self.scaler_backbone = scaler_backbone
        self.tensor_irreps = None

    def transform(self, dataset):
        norms = dataset.norms
        new_norms = self.scaler_backbone.transform(norms)
        scaled_dataset = dataset.apply_norms(new_norms)
        return scaled_dataset

    def inverse_transform(self, data_tensor):
        assert self.tensor_irreps is not None, 'Error: EquivariantScalerWrap has not been set up for inverse transform.'
        tensors = []
        ptr = 0
        for irr in self.tensor_irreps:
            # if irr == 2, that means it's a batch x 3 x 3 tensor and you take 3
            # if irr == 1, that means it's a batch x 3 vector and you take 3
            # if irr == 0, that means it's a scalar and you take 1
            if irr.l == 1:
                length = 3
            else:
                length = 1 + 1 * irr.l
            tensors.append(data_tensor[:, ptr:ptr+length])
            ptr += length
        #assert ptr == data_tensor.shape[1], 'Error. The expected size of tensor and the one passed are different'
        tensors_data = TensorsData(tensors, self.tensor_irreps)
        norms = tensors_data.norms.cpu()
        new_norms = torch.tensor(self.scaler_backbone.inverse_transform(norms)).to(data_tensor.device)
        scaled_dataset = tensors_data.apply_norms(new_norms)
        return torch.cat(scaled_dataset.tensors, dim=1)


    def fit(self, dataset):
        norms = dataset.norms
        self.scaler_backbone.fit(norms)
        if hasattr(self.scaler_backbone, 'coeffs'):
            print('scaler coeffs: ', self.scaler_backbone.coeffs)

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def inverse_setup(self, irrep_list):
        self.tensor_irreps = irrep_list


class MeanScaler:
    def __init__(self, *args, **kwargs):
        self.coeffs = None

    def fit(self, norms):
        # batch x channels x (dims)
        self.coeffs = 1 / torch.mean(norms, dim=0, keepdim=True)

    def transform(self, norms):
        return norms * self.coeffs

    def inverse_transform(self, norms):
        return norms / self.coeffs

    def fit_transform(self, norms):
        self.fit(norms)
        return self.transform(norms)



