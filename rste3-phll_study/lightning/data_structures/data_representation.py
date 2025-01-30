import copy
from dataclasses import dataclass, field
from typing import List

import e3nn
import numpy as np
import torch
from e3nn.io import CartesianTensor

from sklearn.linear_model import LinearRegression
from tqdm import tqdm

import Ofpp
import io
import os
import tempfile



def ofpp_crlf_read(file_path):
    with open(file_path, 'r', newline=None) as file:
        content = file.read()

    normalized_content = content.replace('\r\n', '\n')

    with tempfile.NamedTemporaryFile(delete=False, mode='w+', newline='\n') as temp_file:
        temp_file.write(normalized_content)
        temp_file_path = temp_file.name

    field_data = Ofpp.parse_internal_field(temp_file_path)


    os.remove(temp_file_path)
    return field_data

def foam_get_field_type(file_path):
    field_type = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip().startswith('class'):
                    # Extract the field type from the line
                    field_type = line.split()[-1].strip(';')
                    break
    except IOError:
        print(f"Error opening or reading from {file_path}")

    return field_type



# Import other necessary libraries, e.g., torch, numpy
def anisotropy_tensor(tau):
    return tau - np.identity(3)[None, :, :] * np.einsum('bii -> b', tau)[:, None, None] / 3


def safe_divide(numerator, denominator, safe_value=0):
    # Check for zeros in the denominator
    safe_denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)

    # Perform division
    result = numerator / safe_denominator

    # Replace the results where denominator was zero with a safe value
    safe_result = torch.where(denominator == 0, torch.full_like(result, safe_value), result)

    return safe_result

def optimal_viscosity_params(a_tensor, gradU):
    print('counting optimal viscosity')
    nut_nnls_list = []
    aperp_nnls_list = []
    S_tensor = (gradU + gradU.transpose(0, 2, 1)) / 2  # CHANGE
    # print(S_tensor[0])
    for a, S in tqdm(zip(a_tensor, S_tensor), total=a_tensor.shape[0]):
        X = -2 * np.delete(S.reshape((1, 9)), [3, 6, 7], axis=1).flatten().reshape(-1, 1)
        y = np.delete(a.reshape((1, 9)), [3, 6, 7], axis=1).flatten().reshape(-1, 1)
        reg_nnls = LinearRegression(positive=True, fit_intercept=False)
        reg_nnls.fit(X, y)
        nut_nnls = reg_nnls.coef_[0, 0]
        aperp_nnls = a + 2 * np.repeat(nut_nnls, 9).reshape(nut_nnls.size, 3, 3) * S
        nut_nnls_list.append(nut_nnls)
        aperp_nnls_list.append(aperp_nnls)
        # print('start', a, S, aperp_nnls)
    ev = np.stack(nut_nnls_list)
    ap = np.concatenate(aperp_nnls_list, axis=0)
    return ev, ap


@dataclass
class TensorData:
    # used to load fields that have a record per-node, e.g., velocity field, RST field
    tensor: torch.Tensor
    parity: int = -1
    rank: int = field(init=False)

    # irreps: e3nn.o3.Irreps = field(init=False)

    def __post_init__(self):
        if type(self.tensor) == np.ndarray:
            self.tensor = torch.tensor(self.tensor)
        if self.tensor.dtype == torch.float64:
            self.tensor = torch.tensor(self.tensor, dtype=torch.float32)
        self.infer_dimensionality()

    def infer_dimensionality(self):
        shape = self.shape
        values_per_field = 1
        for d in shape[1:]:
            values_per_field = values_per_field * d
        if values_per_field == 1:
            self.rank = 0
            self.tensor = self.tensor.reshape((-1, 1))
        if values_per_field == 3:
            self.rank = 1
            self.tensor = self.tensor.reshape((-1, 3))
        if values_per_field == 9:
            self.rank = 2
            self.tensor = self.tensor.reshape((-1, 3, 3))
        if self.parity == -1:
            self.parity = self.rank % 2

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def irrep(self):
        parity_symbol = 'e' if self.parity == 0 else 'o'
        return str(self.rank) + parity_symbol

    @property
    def norms(self):
        return torch.norm(self.tensor, dim=tuple(range(1, self.tensor.ndim)))

    def apply_norms(self, norms, inplace=False):
        if not inplace:
            return copy.deepcopy(self).apply_norms(norms, inplace=True)
        else:
            current_norms = self.norms
            new_tensor = (safe_divide(norms, current_norms)).reshape(-1, *[1 for _ in range(self.tensor.ndim - 1)]) * self.tensor
            self.tensor = new_tensor
            return self

    def to_irrep_tensors(self):
        # so far, only scalars, vectors and 2nd rank even tensors are supported
        torch_tensor = torch.tensor(self.tensor)
        if self.rank == 0 and self.parity == 0:
            return [IrrepTensorData(torch_tensor, '0e')]
        if self.rank == 1 and self.parity == 1:
            return [IrrepTensorData(torch_tensor, '1o')]
        if self.rank == 2 and self.parity == 0:
            ct = CartesianTensor('ij')
            spherical_representation = ct.from_cartesian(torch_tensor)
            trace = spherical_representation[:, :1]
            antisymmetric = spherical_representation[:, 1:4]
            symmetric = spherical_representation[:, 4:]
            tensors = [trace, antisymmetric, symmetric]
            irreps = ['0e', '1e', '2e']
            return [IrrepTensorData(t, i) for t, i in zip(tensors, irreps)]
        else:
            raise NotImplementedError(f'Encountered rank {self.rank} and parity {self.parity}')

    @classmethod
    def load(cls, path):
        assert len(path) == 1
        return cls(torch.tensor(np.load(path[0])))

    @classmethod
    def stack(cls, path):
        return cls(torch.stack([torch.tensor(np.load(p)) for p in path], dim=-1))

    @classmethod
    def effective_viscosity(cls, path):
        """ Computes optimal effective viscosity using raw data as in: Structured Deep Neural Network for Turbulence
        Closure Modeling by R. McConkey et al.
        The specified paths should be for: (1) tau; (2) velocity gradient tensor

        The effective viscosity decomposition, as proposed by McConkey, utilizes tau and grad_u both from DNS data
        """
        tau = np.load(path[0])
        grad_u = np.load(path[1])
        a = anisotropy_tensor(tau)
        ev, ap = optimal_viscosity_params(a, grad_u)
        return cls(torch.tensor(ev))

    @classmethod
    def anisotropic_part(cls, path):
        """ Computes the anisotropic part of Reynolds stresses using raw data as in: Structured Deep Neural Network for
        Turbulence Closure Modeling by R. McConkey et al.
        The specified paths should be for: (1) tau; (2) velocity gradient tensor
        """
        tau = np.load(path[0])
        grad_u = np.load(path[1])
        a = anisotropy_tensor(tau)
        ev, ap = optimal_viscosity_params(a, grad_u)
        return cls(torch.tensor(ap))

    @classmethod
    def pope_invariants(cls, path):
        pope_inv = np.load(path[0])
        assert pope_inv.shape[1] == 5
        return [cls(torch.tensor(pope_inv[:, i])) for i in range(pope_inv.shape[1])]

    @classmethod
    def pope_tensors(cls, path):
        pope_tens = np.load(path[0])
        assert pope_tens.shape[1] == 10
        return [cls(torch.tensor(pope_tens[:, i])) for i in range(pope_tens.shape[1])]

    @classmethod
    def foam(cls, path):
        if not isinstance(path, List):
            path = [path]

        field_data = ofpp_crlf_read(path[0])
        field_type = foam_get_field_type(path[0])
        #print('xd')
        if field_type == 'volVectorField':
            data = field_data
            parity = 1
        if field_type == 'volScalarField':
            data = field_data
            parity = 0
        if field_type == 'volSymmTensorField':
            data = field_data[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]]
            parity = 0

        return cls(torch.tensor(data), parity=parity)


    @classmethod
    def foam_symm(cls, path):
        if not isinstance(path, List):
            path = [path]

        field_data = ofpp_crlf_read(path[0])
        field_type = foam_get_field_type(path[0])
        #print('xd')
        if field_type == 'volVectorField':
            raise ValueError(f'Path {path} contains a vector, which cannot be decomposed to symmetric part')
        if field_type == 'volScalarField':
            raise ValueError(f'Path {path} contains a scalar, which cannot be decomposed to symmetric part')
        if field_type == 'volSymmTensorField':
            data = field_data[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]]
            structured_data = data.reshape(-1, 3, 3)
            symm_data = (structured_data + structured_data.transpose(0, 2, 1))/2


            trace = field_data[:, [0, 3, 5]].sum(axis=-1, keepdims=True)
            trace_component = np.eye(3).reshape(1, 3, 3) * trace.reshape(1, 1, 1)
            symm_data -= trace_component

            final_data = symm_data.reshape(-1, 9)
            parity = 0

        return cls(torch.tensor(final_data), parity=parity)

    @classmethod
    def foam_anti(cls, path):
        if not isinstance(path, List):
            path = [path]

        field_data = ofpp_crlf_read(path[0])
        field_type = foam_get_field_type(path[0])
        #print('xd')
        if field_type == 'volVectorField':
            raise ValueError(f'Path {path} contains a vector, which cannot be decomposed to antisymmetric part')
        if field_type == 'volScalarField':
            raise ValueError(f'Path {path} contains a vector, which cannot be decomposed to antisymmetric part')
        if field_type == 'volSymmTensorField':
            data = field_data[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]]
            structured_data = data.reshape(-1, 3, 3)
            symm_data = (structured_data - structured_data.transpose(0, 2, 1))/2
            final_data = symm_data.reshape(-1, 9)
            parity = 0
        return cls(torch.tensor(final_data), parity=parity)

    @classmethod
    def foam_trace(cls, path):
        if not isinstance(path, List):
            path = [path]

        field_data = ofpp_crlf_read(path[0])
        field_type = foam_get_field_type(path[0])
        #print('xd')
        if field_type == 'volVectorField':
            raise ValueError(f'Path {path} contains a vector, which cannot be decomposed to antisymmetric part')
        if field_type == 'volScalarField':
            raise ValueError(f'Path {path} contains a vector, which cannot be decomposed to antisymmetric part')
        if field_type == 'volSymmTensorField':
            data = field_data[:, [0, 3, 5]].sum(axis=-1)
            parity = 0
        return cls(torch.tensor(data), parity=parity)




@dataclass
class TensorsData:
    tensors: List[torch.Tensor]
    irreps: List[e3nn.o3.Irreps]

    def __post_init__(self):
        for i, irr in enumerate(self.irreps):
            if type(irr) == str:
                self.irreps[i] = e3nn.o3.Irrep(irr)

    @classmethod
    def from_list(cls, tensordata: List[TensorData]):
        tens = [td.tensor for td in tensordata]
        irrs = [td.irrep for td in tensordata]
        return cls(tens, irrs)

    def __iter__(self):
        for t, i in zip(self.tensors, self.irreps):
            parity = 0 if str(i)[-1] == 'e' else 1
            yield TensorData(t, parity)

    @property
    def norms(self):
        norms = []
        for tdata in self:
            norms.append(tdata.norms)
        return torch.stack(norms, dim=-1)

    def apply_norms(self, norms, inplace=False):
        if not inplace:
            return copy.deepcopy(self).apply_norms(norms, inplace=True)
        else:
            tensors = []
            for tdata, norm_i in zip(self, range(norms.shape[1])):
                tdata = tdata.apply_norms(norms[:, norm_i])
                tensors.append(tdata.tensor)
            self.tensors = tensors
            return self

    def to_irrepdata(self):
        irrepdata = []
        for tdata in self:
            irrepdata += tdata.to_irrep_tensors()
        return IrrepsTensorData.from_irreptensors(irrepdata)

    def to_concat(self):
        return torch.cat(self.tensors, dim=1)



@dataclass
class IrrepTensorData:
    tensor: torch.Tensor
    irrep: e3nn.o3.Irrep

    def __post_init__(self):
        if type(self.irrep) == str:
            self.irrep = e3nn.o3.Irrep(self.irrep)


@dataclass
class IrrepsTensorData:
    tensor: torch.Tensor
    irreps: e3nn.o3.Irreps

    @property
    def irrep_slices(self):
        ctr = 0
        slices = []
        for l in self.irreps.ls:
            slices.append(slice(ctr, ctr + 2 * l + 1))
            ctr += 2 * l + 1
        return slices

    def __iter__(self):
        pass

    @classmethod
    def from_irreptensors(cls, irreptensors: list):
        tensors = [irt.tensor for irt in irreptensors]
        accumulated_irreps = e3nn.o3.Irreps('')
        for irt in irreptensors:
            accumulated_irreps += irt.irrep
        tensors = torch.cat(tensors, dim=1)
        return cls(tensors, accumulated_irreps)


@dataclass
class SimulationCase:
    name: str
    tensor_fields: dict
    variant_name: str = ''

    def condition_path(self, path_in, extension='.npy'):
        pth = path_in
        self.name = str(self.name)
        self.variant_name = str(self.variant_name)
        if '{case_name}' in pth:
            # print('replacing case name')
            pth = pth.replace('{case_name}', self.name)
        if '{CASE_NAME}' in pth:
            # print('replacing case name in uppercase')
            pth = pth.replace('{CASE_NAME}', self.name.upper())
        if '{case_type}' in pth:
            # print('replacing case type')
            pth = pth.replace('{case_type}', self.variant_name)
        if pth[-len(extension):] != extension:
            pth = pth + extension
        return pth

    def load_tensor(self, path_in, extension='.npy'):
        pth = path_in
        if '{case_name}' in pth:
            # print('replacing case name')
            pth = pth.replace('{case_name}', self.name)
        if '{CASE_NAME}' in pth:
            # print('replacing case name in uppercase')
            pth = pth.replace('{CASE_NAME}', self.name.upper())
        if '{case_type}' in pth:
            # print('replacing case type')
            pth = pth.replace('{case_type}')
        if pth[-len(extension):] != extension:
            pth = pth + extension
        return np.load(pth)

    def load_variant(self, field_list):
        tensors = []
        for field in field_list:
            pth = self.tensor_fields[field]['path']
            field_type = self.tensor_fields[field]['type']
            if type(pth) == str:
                pth = [pth]
            if field_type == 'TensorData.load' or field_type == 'TensorData':
                conditioned_paths = [self.condition_path(pt) for pt in pth]
                tensor_data = TensorData.load(conditioned_paths)
            elif field_type == 'TensorData.stack':
                conditioned_paths = [self.condition_path(pt) for pt in pth]
                tensor_data = TensorData.stack(conditioned_paths)
            elif field_type == 'TensorData.optimal_viscosity':
                conditioned_paths = [self.condition_path(pt) for pt in pth]
                tensor_data = TensorData.effective_viscosity(conditioned_paths)
            elif field_type == 'TensorData.anisotropic_part':
                conditioned_paths = [self.condition_path(pt) for pt in pth]
                tensor_data = TensorData.anisotropic_part(conditioned_paths)
            elif field_type == 'TensorData.pope_invariants':
                conditioned_paths = [self.condition_path(pt) for pt in pth]
                tensor_data = TensorData.pope_invariants(conditioned_paths)
            elif field_type == 'TensorData.pope_tensors':
                conditioned_paths = [self.condition_path(pt) for pt in pth]
                tensor_data = TensorData.pope_tensors(conditioned_paths)
            elif field_type == 'TensorData.foam':
                conditioned_paths = [self.condition_path(pt, extension='') for pt in pth]
                tensor_data = TensorData.foam(conditioned_paths)
            elif field_type == 'TensorData.foam_symm':
                conditioned_paths = [self.condition_path(pt, extension='') for pt in pth]
                tensor_data = TensorData.foam_symm(conditioned_paths)
            elif field_type == 'TensorData.foam_anti':
                conditioned_paths = [self.condition_path(pt, extension='') for pt in pth]
                tensor_data = TensorData.foam_anti(conditioned_paths)
            elif field_type == 'TensorData.foam_trace':
                conditioned_paths = [self.condition_path(pt, extension='') for pt in pth]
                tensor_data = TensorData.foam_trace(conditioned_paths)
            else:
                raise ValueError(f'Tensor field type not recognized, it is {field_type}')
            if type(tensor_data) == list:
                tensors = tensors + tensor_data
            else:
                tensors.append(tensor_data)
        # for t in tensors:
        # print(t.rank, t.parity)
        return TensorsData([t.tensor for t in tensors], [t.irrep for t in tensors])

