from collections import defaultdict

import e3nn
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule


def symmetric_tensor_foam(t, dim='xy'):
    if dim == 'xy':  # return unique values of 2-rank tensor in xy dimensions: xx, xy, yy
        return t[:, [0, 0, 1], [0, 1, 1]]
    if dim == 'xyz':  # return unique values of 2-rank tensor in xyz dimensions: xx, xy, xz, yy, yz, zz
        return t[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    if dim == 's':  # if values are scalar (effective viscosity), just pass them on
        return t


def symmetric_tensor_unique(t, dim='xy'):
    if dim == 'xy':  # return unique values of 2-rank tensor in xy dimensions: xx, yy, xy
        return t[:, [0, 1, 0], [0, 1, 1]]
    if dim == 'xyz':  # return unique values of 2-rank tensor in xyz dimensions: xx, yy, zz, xy, xz, yz
        return t[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    if dim == 's':  # if values are scalar (effective viscosity), just pass them on
        return t


class TurbulenceClosureWrap(LightningModule):
    def __init__(self, training_cfg, nn, output_format, irreps_in, loss_weights, *args, **kwargs):
        super(TurbulenceClosureWrap,
              self).__init__()  # This will save model_cfg, training_cfg, and irreps_in to self.hparams
        self.training_cfg = training_cfg
        self.irreps_in = irreps_in
        self.output_format = output_format
        if self.output_format == 'tensor':
            self.irreps_out = '1x0e + 1x2e'
        elif self.output_format == 'scalar':
            self.irreps_out = '1x0e'
        elif self.output_format == 'vector':
            self.irreps_out = '1x1o'
        else:
            raise AttributeError(
                'Output format not recognized. Each output format demands different symbolic postprocessing')

        self.ct = e3nn.io.CartesianTensor('ij=ji')
        self.lct = e3nn.io.CartesianTensor('ij=ij')
        self.rtp = None

        # Note that irreps_in, target_dim etc. are now accessible directly via self.hparams
        self.nn = hydra.utils.instantiate(nn,
                                          irreps_in=self.irreps_in,
                                          irreps_out=self.irreps_out)

        # Accessing other config parameters via self.hparams
        self.output_format = output_format
        self.loss_weights = loss_weights
        self.lr = training_cfg.lr

        # Some initial states
        self.first_step = True
        self.data_scaler = None
        self.label_scaler = None
        self.rtp = self.ct.reduced_tensor_products(torch.tensor([0., 1., 2.]))
        self.lrtp = self.lct.reduced_tensor_products(torch.tensor([0., 1., 2.]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, attributes):
        res = self.nn(attributes)
        if self.output_format == 'tensor':
            return self.ct.to_cartesian(res, rtp=self.rtp)
        else:
            return res

    def predict(self, batch, dimensions='xyz',
        as_numpy=True, non_negative=False, ignore_nondiag_z=True,
                condition_trace=True):
        attributes = batch.x
        out = self(attributes)
        out = out.detach().cpu()
        # norms = out.norm(dim=(1, 2))
        # cart_out = self.label_scaler.inverse_transform(out)
        scaled_out = self.label_scaler.inverse_transform(out)
        if non_negative:
            scaled_out = torch.maximum(scaled_out, torch.tensor(0.))
        if self.output_format == 'tensor':
            if ignore_nondiag_z and dimensions == 'xy':
                scaled_out[:, 0, 2] *= 0
                scaled_out[:, 1, 2] *= 0
                scaled_out[:, 2, 1] *= 0
                scaled_out[:, 2, 1] *= 0
            if condition_trace and dimensions == 'xy':
                scaled_out[:, 2, 2] = -scaled_out[:, 0, 0] - scaled_out[:, 1, 1]
            dimensional_out = symmetric_tensor_foam(scaled_out, dim='xyz')
        elif self.output_format == 'vector':
            if ignore_nondiag_z:
                dimension_weights = [1, 1, 1] if dimensions == 'xyz' else [1, 1, 0]
                dimensional_out = scaled_out * torch.tensor(dimension_weights,
                                                            dtype=scaled_out.dtype, device=scaled_out.device)[None, :]
        elif self.output_format == 'scalar':
            dimensional_out = scaled_out
        else:
            raise ValueError



        if as_numpy:
            return dimensional_out.detach().cpu().numpy()
        else:
            return dimensional_out

    def predict_from_loader(self, loader, device, dimensions='xyz', as_numpy=True, non_negative=False, condition_trace=True,
                            ignore_nondiag_z=True):
        collected_outputs = []
        for batch in loader:
            batch.to(device)
            collected_outputs.append(self.predict(batch, dimensions=dimensions, as_numpy=as_numpy,
                                                  non_negative=non_negative, condition_trace=condition_trace,
                                                  ignore_nondiag_z=ignore_nondiag_z))

        if as_numpy:
            return np.concatenate(collected_outputs, axis=0)
        return torch.cat(collected_outputs, dim=0)

    def shared_step(self, batch, prefix='unspecified'):
        # Initial setup that needs to happen regardless of the output format
        if self.first_step:
            if self.output_format == 'tensor':
                self.rtp = self.ct.reduced_tensor_products(batch.x)
                self.lrtp = self.lct.reduced_tensor_products(batch.y)
            self.first_step = False

        if self.output_format == 'tensor':
            return self.tensor_step(batch, prefix)
        elif self.output_format == 'vector':
            return self.vector_step(batch, prefix)

        elif self.output_format == 'scalar':
            return self.scalar_step(batch, prefix)

    def scalar_step(self, batch, prefix='unspecified'):
        attributes, labels = batch.x, batch.y

        # Perform operations for scalar output
        out = self(attributes)
        scaled_out = self.label_scaler.inverse_transform(out.detach())
        losses = defaultdict(lambda: torch.tensor(0.0, device=self.device))
        losses['loss_scalar'] = F.mse_loss(out, labels)
        scaled_labels = self.label_scaler.inverse_transform(labels.detach())

        # Calculate original losses for logging
        losses['original_loss_scalar'] = F.mse_loss(scaled_out, scaled_labels)
        return self.log_losses(losses, prefix)

    def vector_step(self, batch, prefix='unspecified'):
        attributes, labels = batch.x, batch.y

        out = self(attributes)
        scaled_out = self.label_scaler.inverse_transform(out.detach())
        losses = defaultdict(lambda: torch.tensor(0.0, device=self.device))
        losses['loss_xyz'] = F.mse_loss(out, labels)
        losses['loss_xy'] = F.mse_loss(out[..., :2], labels[..., :2])
        scaled_labels = self.label_scaler.inverse_transform(labels.detach())

        losses['original_loss_xyz'] = F.mse_loss(scaled_out, scaled_labels)
        losses['original_loss_xy'] = F.mse_loss(scaled_out[..., :2], scaled_labels[..., :2])
        return self.log_losses(losses, prefix)

    def tensor_step(self, batch, prefix='unspecified'):
        attributes, labels = batch.x, batch.y

        # Perform operations for tensor output
        out = self(attributes)
        scaled_out = self.label_scaler.inverse_transform(out.detach())
        losses = defaultdict(lambda: torch.tensor(0.0, device=self.device))
        cart_labels = self.lct.to_cartesian(labels, rtp=self.lrtp)
        losses['loss_xyz'] = F.mse_loss(symmetric_tensor_unique(out, 'xyz'),
                                        symmetric_tensor_unique(cart_labels, 'xyz'))
        losses['loss_xy'] = F.mse_loss(symmetric_tensor_unique(out, 'xy'),
                                       symmetric_tensor_unique(cart_labels, 'xy'))
        scaled_labels = self.label_scaler.inverse_transform(cart_labels.detach())

        # Calculate original losses for logging
        losses['original_loss_xyz'] = F.mse_loss(symmetric_tensor_unique(scaled_out, 'xyz'),
                                                 symmetric_tensor_unique(scaled_labels, 'xyz'))
        losses['original_loss_xy'] = F.mse_loss(symmetric_tensor_unique(scaled_out, 'xy'),
                                                symmetric_tensor_unique(scaled_labels, 'xy'))
        return self.log_losses(losses, prefix)

    def log_losses(self, losses, prefix):
        self.log_dict(
            {f'{prefix}_{k}': v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        loss = torch.sum(torch.stack([losses[k] * self.loss_weights[k] for k in self.loss_weights.keys()]))
        losses['loss'] = loss
        self.log(
            f'{prefix}_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, prefix='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, prefix='val')

    def test_step(self, batch, batch_idx):
        attributes, labels = batch.x, batch.y
        prefix = 'test'
        # this one needs to be saved so as not to reinitialize rtp, causing slowdown
        if self.first_step:
            if self.output_format == 'tensor':
                self.rtp = self.ct.reduced_tensor_products(attributes)
                self.lrtp = self.lct.reduced_tensor_products(labels)
            self.first_step = False

        out = self(attributes)
        losses = defaultdict(lambda: torch.tensor(0.0, device=self.device))
        if self.output_format == 'tensor':
            cart_labels = self.lct.to_cartesian(labels, rtp=self.lrtp)
            losses['loss_xyz'] = F.mse_loss(symmetric_tensor_unique(out, 'xyz'),
                                            symmetric_tensor_unique(cart_labels, 'xyz'))
            losses['loss_xy'] = F.mse_loss(symmetric_tensor_unique(out, 'xy'),
                                           symmetric_tensor_unique(cart_labels, 'xy'))

        if self.output_format == 'vector':
            losses['loss_xyz'] = F.mse_loss(out, labels)
            losses['loss_xy'] = F.mse_loss(out[..., :2], labels[..., :2])
            cart_labels = labels

        if self.output_format == 'scalar':
            losses['loss_scalar'] = F.mse_loss(out, labels)
            cart_labels = labels

        loss = torch.sum(torch.stack([losses[k] * self.loss_weights[k] for k in self.loss_weights.keys()]))
        losses['loss'] = loss

        original_out = self.label_scaler.inverse_transform(out)
        original_labels = self.label_scaler.inverse_transform(cart_labels)
        if self.output_format == 'tensor':
            cart_labels = self.lct.to_cartesian(labels, rtp=self.lrtp)
            losses['original_loss_xyz'] = F.mse_loss(symmetric_tensor_unique(original_out, 'xyz'),
                                                     symmetric_tensor_unique(original_labels, 'xyz'))
            losses['original_loss_xy'] = F.mse_loss(symmetric_tensor_unique(original_out, 'xy'),
                                                    symmetric_tensor_unique(original_labels, 'xy'))

            losses['XX loss'] = F.mse_loss(original_out[:, 0, 0], original_labels[:, 0, 0])
            losses['YY loss'] = F.mse_loss(original_out[:, 1, 1], original_labels[:, 1, 1])
            losses['XY loss'] = F.mse_loss(original_out[:, 0, 1], original_labels[:, 0, 1])
            losses['YX loss sanity check'] = F.mse_loss(original_out[:, 1, 0], original_labels[:, 1, 0])

        if self.output_format == 'vector':
            losses['original_loss_xyz'] = F.mse_loss(original_out, original_labels)
            losses['original_loss_xy'] = F.mse_loss(original_out[..., :2], original_labels[..., :2])

            losses['X loss'] = F.mse_loss(original_out[:, 0], original_labels[:, 0])
            losses['Y loss'] = F.mse_loss(original_out[:, 1], original_labels[:, 1])
            losses['Z loss'] = F.mse_loss(original_out[:, 2], original_labels[:, 2])

        if self.output_format == 'scalar':
            losses['original_loss_scalar'] = F.mse_loss(original_out, original_labels)

        self.log_dict(
            {f'{prefix}_{k}': v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            f'{prefix}_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def original_labels(self, out, labels):
        cart_out, cart_labels = self.ct.to_cartesian(out), self.lct.to_cartesian(labels)


