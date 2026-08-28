import warnings

import numpy as np
import torch
from torch import nn

from .. import activations
from .base import Encoder


def select_decov_target_vecs(feature_vecs, whitener, decorrelation_on_raw_features):
    """
    Picks which feature vectors the DeCov / variance-floor penalty should be computed on:
    the raw (pre-whitening) readout output when `decorrelation_on_raw_features` is set and a
    batch-mode whitener produced them this forward call, otherwise `feature_vecs` as-is
    (already raw if whitener is None or in 'ema' mode).
    """
    target_vecs = feature_vecs
    if decorrelation_on_raw_features and whitener is not None and whitener.mode == "batch":
        target_vecs = whitener.last_raw_feature_vecs
    return target_vecs


def decov_variance_floor_loss(flat_target_vecs, decov_weight, variance_floor_weight, variance_floor_gamma):
    """
    Computes the DeCov (off-diagonal covariance) and variance-floor (per-channel std hinge)
    penalties from a single (n, c) matrix of feature vectors, with full autograd through the
    covariance. `flat_target_vecs` may pool samples from a single session's batch or be a
    concatenation across several sessions' batches -- the statistics (mean, covariance) are
    simply computed over whatever samples are present, so pooling across sessions couples
    their gradients through the shared mean/covariance.
    """
    device = flat_target_vecs.device
    variance_floor_loss = torch.zeros((), device=device)
    decov_loss = torch.zeros((), device=device)

    if variance_floor_weight <= 0 and decov_weight <= 0:
        return variance_floor_loss, decov_loss

    fv = flat_target_vecs
    fv_centered = fv - fv.mean(dim=0, keepdim=True)
    n, c = fv.shape
    cov = fv_centered.T @ fv_centered / (n - 1)

    if variance_floor_weight > 0:
        std = cov.diagonal().clamp_min(1e-12).sqrt()
        variance_floor_loss = variance_floor_weight * torch.relu(variance_floor_gamma - std).mean()

    if decov_weight > 0:
        off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        decov_loss = decov_weight * off_diag_sq / (c * (c - 1))

    return variance_floor_loss, decov_loss


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
                target_vecs = select_decov_target_vecs(
                    feature_vecs, self.whitener, self.decorrelation_on_raw_features
                )
                self.last_variance_floor_loss, self.last_decov_loss = decov_variance_floor_loss(
                    target_vecs.flatten(0, 1), self.decov_weight, self.variance_floor_weight, self.variance_floor_gamma
                )

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

    def forward_raw(
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
        **kwargs
    ):
        """
        Runs core (+ perspective/shifter) and the readout's raw, pre-whitening per-neuron
        feature extraction for `data_key` (whitener is forced off here), returning the core
        output and resolved shift alongside those raw feature vectors. Pair with
        `forward_from_features` to finish the forward pass once the raw feature vectors --
        possibly from several sessions -- have been whitened externally (e.g. jointly).
        """
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

        _, raw_feature_vecs = self.readout(x, data_key=data_key, shift=shift, whitener=None, **kwargs)

        return x, raw_feature_vecs, shift

    def forward_from_features(
        self,
        core_out,
        *,
        data_key=None,
        whitened_feature_vecs,
        shift=None,
        behavior=None,
        return_vec=False,
        **kwargs
    ):
        """
        Finishes a forward pass started by `forward_raw`: applies the readout's final per-neuron
        combination to the given (externally whitened) feature vectors, then the modulator and
        nonlinearity -- mirroring the tail of `forward`. Does not touch
        last_variance_floor_loss/last_decov_loss; the caller is expected to compute those itself
        from whatever feature vectors it pooled across sessions.
        """
        x, feature_vecs = self.readout(
            core_out, data_key=data_key, shift=shift, external_whitened_feature_vecs=whitened_feature_vecs, **kwargs
        )

        x = x[None, ...] if len(x.shape) == 1 else x  # keep dimensions if only one image was passed

        if self.modulator:
            if behavior is None:
                raise ValueError("behavior is not given")
            x = self.modulator[data_key](x, behavior=behavior)

        if self.nonlinearity_type == "elu":
            out = self.nonlinearity_fn(x + self.offset) + 1
        else:
            out = self.nonlinearity_fn(x)

        return (out, feature_vecs) if return_vec else out

    def predict_mean(self, x, *args, data_key=None, **kwargs):
        return self.forward(x, *args, data_key=data_key, **kwargs)

    def predict_variance(self, x, *args, data_key=None, **kwargs):
        return self.forward(x, *args, data_key=data_key, **kwargs)
