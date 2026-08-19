import warnings

import numpy as np
import torch
from torch import nn

from .. import activations
from .base import Encoder


class FiringRateEncoder(Encoder):
    def __init__(
        self,
        core,
        whitener,
        readout,
        *,
        perspective=None,
        shifter=None,
        modulator=None,
        elu_offset=0.0,
        nonlinearity_type="elu",
        nonlinearity_config=None,
        variance_floor_weight=0.0,
        variance_floor_gamma=1.0,
        decov_weight=0.0,
        decorrelation_on_raw_features=False,
    ):
        """
        An Encoder that wraps the core, readout and optionally a shifter amd modulator into one model.
        The output is one positive value that can be interpreted as a firing rate, for example for a Poisson distribution.
        Args:
            core (nn.Module): Core model. Refer to neuralpredictors.layers.cores
            readout (nn.ModuleDict): MultiReadout model. Refer to neuralpredictors.layers.readouts
            elu_offset (float): Offset value in the final elu non-linearity. Defaults to 0.
            shifter (optional[nn.ModuleDict]): Shifter network. Refer to neuralpredictors.layers.shifters. Defaults to None.
            modulator (optional[nn.ModuleDict]): Modulator network. Modulator networks are not implemented atm (24/06/2021). Defaults to None.
            nonlinearity (str): Non-linearity type to use. Defaults to 'elu'.
            nonlinearity_config (optional[dict]): Non-linearity configuration. Defaults to None.
            variance_floor_weight (float): weight of a per-channel variance floor hinge loss on feature_vecs,
                penalizing channels whose std falls below variance_floor_gamma. 0 disables it.
            variance_floor_gamma (float): target minimum per-channel std for the variance floor loss.
            decov_weight (float): weight of an off-diagonal covariance penalty (DeCov) on feature_vecs,
                penalizing correlation between channels. 0 disables it.
            decorrelation_on_raw_features (bool): when whitener.mode == 'batch', apply the variance
                floor / DeCov penalties to the pre-whitening feature_vecs (the readout's raw sampled
                core output) instead of the whitened ones. Has no effect if whitener is None or in
                'ema' mode, since feature_vecs is already the raw features in that case.
        """
        super().__init__()
        self.core = core
        self.whitener = whitener
        self.readout = readout
        self.perspective = perspective
        self.shifter = shifter
        self.modulator = modulator
        self.offset = elu_offset
        self.variance_floor_weight = variance_floor_weight
        self.variance_floor_gamma = variance_floor_gamma
        self.decov_weight = decov_weight
        self.decorrelation_on_raw_features = decorrelation_on_raw_features
        self.last_variance_floor_loss = 0.0
        self.last_decov_loss = 0.0

        if nonlinearity_type != "elu" and not np.isclose(elu_offset, 0.0):
            warnings.warn("If `nonlinearity_type` is not 'elu', `elu_offset` will be ignored")
        if nonlinearity_type == "elu":
            self.nonlinearity_fn = nn.ELU()
        elif nonlinearity_type == "identity":
            self.nonlinearity_fn = nn.Identity()
        else:
            self.nonlinearity_fn = activations.__dict__[nonlinearity_type](
                **nonlinearity_config if nonlinearity_config else {}
            )
        self.nonlinearity_type = nonlinearity_type

    def forward(
        self,
        inputs,
        *args,
        targets=None,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        return_vec=False,
        **kwargs
    ):
        x = inputs

        if self.perspective:
            if self.shifter:
                raise ValueError("both perspective and shifter cannot be present together, only one should be chosen")
            
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            
            x = self.perspective[data_key](x, pupil_center)

        x = self.core(x)
        if detach_core:
            x = x.detach()

        if self.shifter and pupil_center is not None and shift is None:
            shift = self.shifter[data_key](pupil_center, trial_idx)

        x, feature_vecs = self.readout(x, data_key=data_key, shift=shift, whitener=self.whitener, **kwargs)

        if self.whitener and self.training and self.whitener.mode == 'ema':
            self.whitener.update(feature_vecs)

        if self.training:
            device = feature_vecs.device
            self.last_variance_floor_loss = torch.zeros((), device=device)
            self.last_decov_loss = torch.zeros((), device=device)

            if self.variance_floor_weight > 0 or self.decov_weight > 0:
                target_vecs = feature_vecs
                if (
                    self.decorrelation_on_raw_features
                    and self.whitener is not None
                    and self.whitener.mode == 'batch'
                ):
                    target_vecs = self.whitener.last_raw_feature_vecs

                fv = target_vecs.flatten(0, 1)
                fv_centered = fv - fv.mean(dim=0, keepdim=True)
                n, c = fv.shape
                cov = fv_centered.T @ fv_centered / (n - 1)

                if self.variance_floor_weight > 0:
                    std = cov.diagonal().clamp_min(1e-12).sqrt()
                    self.last_variance_floor_loss = self.variance_floor_weight * torch.relu(
                        self.variance_floor_gamma - std
                    ).mean()

                if self.decov_weight > 0:
                    off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
                    self.last_decov_loss = self.decov_weight * off_diag_sq / (c * (c - 1))

        x = x[None, ...] if len(x.shape) == 1 else x  # keep dimensions if only one image was passed

        if self.modulator:
            if behavior is None:
                raise ValueError("behavior is not given")
            x = self.modulator[data_key](x, behavior=behavior)

        if self.nonlinearity_type == "elu":
            if return_vec:
                return self.nonlinearity_fn(x + self.offset) + 1, feature_vecs
            else: 
                return self.nonlinearity_fn(x + self.offset) + 1
        else:
            if return_vec:
                return self.nonlinearity_fn(x), feature_vecs
            else:
                return self.nonlinearity_fn(x)

    def predict_mean(self, x, *args, data_key=None, **kwargs):
        return self.forward(x, *args, data_key=data_key, **kwargs)

    def predict_variance(self, x, *args, data_key=None, **kwargs):
        return self.forward(x, *args, data_key=data_key, **kwargs)
