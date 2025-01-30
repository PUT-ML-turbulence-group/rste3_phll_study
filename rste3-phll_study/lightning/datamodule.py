from typing import List

import hydra
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from lightning.data_structures.data_representation import TensorsData, SimulationCase


# Import other necessary libraries, e.g., torch, numpy


class SimpleDataset(Dataset):
    """
    This is a dataset that accepts kwargs, similar to torch_geometric.Data, yet doesn't implement
    torch_geometric's framework for graphs. It's designed to simply access the variables as keywords
    (e.g. batch.x), while maintaining the simplicity of classes such as TensorDatabase.
    """

    def __init__(self, **kwargs):
        self.data = kwargs
        self._check_tensor_lengths()

    def _check_tensor_lengths(self):
        lengths = [tensor.size(0) for tensor in self.data.values()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All tensors must have the same size in the first dimension")

    def __len__(self):
        return next(iter(self.data.values())).size(0)

    def __getitem__(self, idx):
        return SimpleData({name: tensor[idx] for name, tensor in self.data.items()})


class SimpleData(dict):
    """ A simple custom class to support SimpleDataset with dot notation, akin to torch_geometric's Data object. """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value

    def to(self, device):
        for k, v in self.items():
            self[k] = v.to(device)
        return self

class IrrepDataModule(pl.LightningDataModule):
    def __init__(self, datasets, datafiles, attributes, labels, data_scaler, label_scaler, batch_size=64,
                 extra_attributes=None, extra_scalers=None):
        super().__init__()
        self.datasets = datasets
        self.datafiles = datafiles
        self.attributes = attributes
        self.labels = labels
        self.data_scaler = data_scaler
        self.label_scaler = label_scaler
        self.batch_size = batch_size
        self.irreps_x = None
        self.irreps_y = None

        self.extra_attributes = extra_attributes
        self.setup()


    def setup(self, stage=None):
        self.train_dataset = self.create_dataset(self.datasets.train)  ###
        #train_x_norms = self.train_dataset['x'].norms
        #train_y_norms = self.train_dataset['y'].norms
        #self.data_scaler.fit(train_x_norms)
        #self.label_scaler.fit(train_y_norms)
        self.data_scaler.fit(self.train_dataset['x'])
        self.data_scaler.inverse_setup(self.train_dataset['x'].irreps)
        self.label_scaler.fit(self.train_dataset['y'])
        self.label_scaler.inverse_setup(self.train_dataset['y'].irreps)

        # extra
        if self.extra_attributes:
            for ex_att in self.extra_attributes:
                ex_att.scaler.fit(self.train_dataset[ex_att.name])
                ex_att.scaler.inverse_setup(self.train_dataset[ex_att.name].irreps)



        self.val_dataset = self.create_dataset(self.datasets.val)
        self.test_dataset = self.create_dataset(self.datasets.test)


    def create_dataset(self, dataset_config):  ###
        dataset = {}
        accumulated_x = []
        accumulated_y = []

        # extra
        if self.extra_attributes:
            accumulated_extra = {ex_att.name: [] for ex_att in self.extra_attributes}

        for case in dataset_config.cases:
            case_name = case.name
            case_variant = case.variant
            case_instance = SimulationCase(case_name, self.datafiles.fields,
                                           variant_name=case_variant)
            tensor_data_list = case_instance.load_variant(self.attributes)
            tensor_label_list = case_instance.load_variant(self.labels)
            tensors_data = TensorsData.from_list(tensor_data_list)
            tensors_labels = TensorsData.from_list(tensor_label_list)
            accumulated_x.append(tensors_data)
            accumulated_y.append(tensors_labels)

            # extra
            if self.extra_attributes:
                for ex_att in self.extra_attributes:
                    ex_tensor_data_list = case_instance.load_variant(ex_att.field_names)
                    ex_data = TensorsData.from_list(ex_tensor_data_list)
                    accumulated_extra[self.ex_att.name].append(ex_data)
        dataset['x'] = self.merge_case_datapoints(accumulated_x)
        dataset['y'] = self.merge_case_datapoints(accumulated_y)

        # extra
        if self.extra_attributes:
            for ex_att in range(len(self.extra_attributes)):
                dataset[ex_att.name] = self.merge_case_datapoints(accumulated_extra[ex_att.name])


        return dataset

    @staticmethod
    def merge_case_datapoints(casedata_list: List[TensorsData]):
        # take the first simulation case and iterate over fields
        tensors = []
        for field_i, _ in enumerate(casedata_list[0]):
            # new field is a stack of all datapoint in the cases
            tensors.append(torch.cat([cd.tensors[field_i] for cd in casedata_list], dim=0))
        return TensorsData(tensors, casedata_list[0].irreps)

    def general_dataloader(self, dataset, shuffle=True):
        #scaled_x = self.scale_dataset(dataset['x'], self.data_scaler)
        #scaled_y = self.scale_dataset(dataset['y'], self.label_scaler)
        scaled_x = self.data_scaler.transform(dataset['x'])
        scaled_y = self.label_scaler.transform(dataset['y'])
        irrepdata_x = scaled_x.to_irrepdata()
        irrepdata_y = scaled_y.to_irrepdata()
        if self.irreps_x is None:
            self.irreps_x = irrepdata_x.irreps
        if self.irreps_y is None:
            self.irreps_y = irrepdata_y.irreps
        else:
            assert self.irreps_x == irrepdata_x.irreps, 'Error! The dataset irreps are different from the ones seen before'
        tensor_dataset = SimpleDataset(x=irrepdata_x.tensor, y=irrepdata_y.tensor)
        return DataLoader(tensor_dataset, self.batch_size, shuffle=shuffle)

    def train_dataloader(self):
        # Return the dataloader for training
        return self.general_dataloader(self.train_dataset)

    def val_dataloader(self):
        # Return the dataloader for validation
        return self.general_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        # Return the dataloader for testing
        return self.general_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        return self.general_dataloader(self.test_dataset, shuffle=False)

class PopeDataModule(pl.LightningDataModule):
    def __init__(self, datasets, datafiles, attributes, basis, labels, data_scaler, basis_scaler, label_scaler, batch_size=64,
                 extra_attributes=None, extra_scalers=None):
        super().__init__()
        self.datasets = datasets
        self.datafiles = datafiles
        self.attributes = attributes
        self.basis = basis
        self.labels = labels
        self.data_scaler = data_scaler
        self.basis_scaler = basis_scaler
        self.label_scaler = label_scaler
        self.batch_size = batch_size
        self.irreps_x = None
        self.irreps_b = None
        self.irreps_y = None


        self.setup()

    def setup(self, stage=None):
        self.train_dataset = self.create_dataset(self.datasets.train)  ###
        # train_x_norms = self.train_dataset['x'].norms
        # train_y_norms = self.train_dataset['y'].norms
        # self.data_scaler.fit(train_x_norms)
        # self.label_scaler.fit(train_y_norms)
        self.data_scaler.fit(self.train_dataset['x'])
        self.data_scaler.inverse_setup(self.train_dataset['x'].irreps)
        self.basis_scaler.fit(self.train_dataset['b'])
        self.basis_scaler.inverse_setup(self.train_dataset['b'].irreps)
        self.label_scaler.fit(self.train_dataset['y'])
        self.label_scaler.inverse_setup(self.train_dataset['y'].irreps)

        self.val_dataset = self.create_dataset(self.datasets.val)
        self.test_dataset = self.create_dataset(self.datasets.test)

    def create_dataset(self, dataset_config):  ###
        dataset = {}
        accumulated_x = []
        accumulated_b = []
        accumulated_y = []

        for case in dataset_config.cases:
            case_name = case.name
            case_variant = case.variant
            case_instance = SimulationCase(case_name, self.datafiles.fields,
                                           variant_name=case_variant)
            tensor_data_list = case_instance.load_variant(self.attributes)
            tensor_basis_list = case_instance.load_variant(self.basis)
            tensor_label_list = case_instance.load_variant(self.labels)
            tensors_data = TensorsData.from_list(tensor_data_list)
            tensors_basis = TensorsData.from_list(tensor_basis_list)
            tensors_labels = TensorsData.from_list(tensor_label_list)
            accumulated_x.append(tensors_data)
            accumulated_b.append(tensors_basis)
            accumulated_y.append(tensors_labels)


        dataset['x'] = self.merge_case_datapoints(accumulated_x)
        dataset['b'] = self.merge_case_datapoints(accumulated_b)
        dataset['y'] = self.merge_case_datapoints(accumulated_y)
        return dataset

    @staticmethod
    def merge_case_datapoints(casedata_list: List[TensorsData]):
        # take the first simulation case and iterate over fields
        tensors = []
        for field_i, _ in enumerate(casedata_list[0]):
            # new field is a stack of all datapoint in the cases
            tensors.append(torch.cat([cd.tensors[field_i] for cd in casedata_list], dim=0))
        return TensorsData(tensors, casedata_list[0].irreps)

    def general_dataloader(self, dataset):
        # scaled_x = self.scale_dataset(dataset['x'], self.data_scaler)
        # scaled_y = self.scale_dataset(dataset['y'], self.label_scaler)
        scaled_x = self.data_scaler.transform(dataset['x'])
        scaled_b = self.basis_scaler.transform(dataset['b'])
        scaled_y = self.label_scaler.transform(dataset['y'])
        cartdata_x = torch.tensor(scaled_x.to_concat(), dtype=torch.float32)
        cartdata_y = scaled_y.to_concat()
        cartdata_b = dataset['b'].to_concat()

        tensor_dataset = SimpleDataset(x=cartdata_x, b=cartdata_b, y=cartdata_y)
        return DataLoader(tensor_dataset, self.batch_size, shuffle=True)

    def train_dataloader(self):
        # Return the dataloader for training
        return self.general_dataloader(self.train_dataset)

    def val_dataloader(self):
        # Return the dataloader for validation
        return self.general_dataloader(self.val_dataset)

    def test_dataloader(self):
        # Return the dataloader for testing
        return self.general_dataloader(self.test_dataset)








