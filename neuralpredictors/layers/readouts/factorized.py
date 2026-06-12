import warnings

import math
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from .base import Readout


class RetinotopySpatial(nn.Module):
    def __init__(self, source_grid, out_shape, factorize_spatial, hidden_features=20, hidden_layers=0):
        super().__init__()

        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer("source_grid", torch.from_numpy(source_grid.astype(np.float32)))
        self.out_shape = out_shape
        self.factorize_spatial = factorize_spatial

        def get_mlp(out_dim):
            in_dim = source_grid.shape[1]
            layers = []
            for hidden_dim in [hidden_features] * hidden_layers:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ELU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            return nn.Sequential(*layers)

        n, h, w = self.out_shape
        if self.factorize_spatial:
            self.h_mlp = get_mlp(h)
            self.w_mlp = get_mlp(w)
        else:
            self.mlp = get_mlp(h * w)

    def forward(self):
        n, h, w = self.out_shape
        
        if self.factorize_spatial:
            h_spatial = self.h_mlp(self.source_grid).unsqueeze(2)
            w_spatial = self.w_mlp(self.source_grid).unsqueeze(1)
        
            spatial = (
                h_spatial.expand(n, h, w) 
                * w_spatial.expand(n, h, w)
            )
        else:
            spatial = self.mlp(self.source_grid).view(n, h, w)
        
        return spatial

def real_fourier_basis(H, W, max_freq):
    y = torch.arange(H)
    x = torch.arange(W)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    basis = []
    for u in range(-max_freq, max_freq + 1):
        for v in range(-max_freq, max_freq + 1):
            phase = 2 * torch.pi * (
                u * Y  + v * X 
            ) / max(H, W)

            # also get rid of cosine for 0, 0 as we use softmax which is logit shift invariant
            if not (u == 0 and v == 0):
                basis.append(torch.cos(phase))
                basis.append(torch.sin(phase))

    basis = torch.stack(basis)
    mask = torch.zeros(basis.shape[0], dtype=bool)
    for i in range(basis.shape[0]):
        repeat = (basis[i, None] == basis[mask]).all(dim=(1, 2)).any()
        inverse = (basis[i, None] == -basis[mask]).all(dim=(1, 2)).any()

        if not repeat and not inverse:
            mask[i] = True
    return basis[mask]

class FourierRetinotopy(nn.Module):
    def __init__(self, source_grid, out_shape, max_freq=3, hidden_features=20, hidden_layers=0):
        super().__init__()

        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer("source_grid", torch.from_numpy(source_grid.astype(np.float32)))
        self.out_shape = out_shape

        n, h, w = self.out_shape
        self.register_buffer("basis", real_fourier_basis(h, w, max_freq))
        k = self.basis.shape[0]

        in_dim = source_grid.shape[1]
        layers = []
        for hidden_dim in [hidden_features] * hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, k))
        self.mlp = nn.Sequential(*layers)

    def forward(self):
        k = self.basis.shape[0]
        coeffs = self.mlp(self.source_grid)  # n, k
        logits = torch.einsum("nk,khw->nhw", coeffs, self.basis) / math.sqrt(k)

        return logits
    

def shift_feature_maps(x, shifts):
    """
    x:      (B, C, H, W)
    shifts: (B, 2) in normalized coords [-1,1]

    returns:
        shifted x: (B, C, H, W)
    """

    B, C, H, W = x.shape
    device = x.device

    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack((xx, yy), dim=-1)  # (H,W,2)

    sampling_grid = (
        base_grid[None]
        - shifts[:, None, None]
    )  # (B,H,W,2)

    return F.grid_sample(
        x,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

class FullFactorized2d(Readout):
    """
    Factorized fully connected layer. Weights are a sum of outer products between a spatial filter and a feature vector.
    """

    def __init__(
        self,
        in_shape,  # channels, height, width
        outdims,
        bias,
        normalize=True,
        init_noise=1e-3,
        constrain_pos=False,
        positive_weights=False,
        positive_spatial=False,
        shared_features=None,
        mean_activity=None,
        spatial_reg_weight=None,
        feature_reg_weight=None,
        gamma_readout=None,
        temperature=None,
        factorize_spatial=False,
        regularizer_type="l1",
        gamma_sigma=0.1,
        source_grid=None,
        retinotopy_spatial=None,
        fourier_spatial=True,
        fourier_max_freq=4,
        whitener=None,
        **kwargs,
    ):
        """

        Args:
            in_shape: batch, channels, height, width (batch could be arbitrary)
            outdims: number of neurons to predict
            bias: if True, bias is used
            normalize: if True, normalizes the spatial mask using l2 norm
            init_noise: the std for readout  initialisation
            constrain_pos: if True, negative values in the spatial mask and feature readout are clamped to 0
            positive_weights: if True, negative values in the feature readout are turned into 0
            positive_spatial: if True, spatial readout mask values are restricted to be positive by taking the absolute values
            shared_features: if True, uses a copy of the features from somewhere else
            mean_activity: the mean for readout  initialisation
            spatial_and_feature_reg_weight: lagrange multiplier (constant) for L1 penalty,
                the bigger the number, the stronger the penalty
            gamma_readout: depricated, use spatial_and_feature_reg_weight instead
            **kwargs:
        """

        super().__init__()

        h, w = in_shape[1:]  # channels, height, width
        self.in_shape = in_shape
        self.outdims = outdims
        self.positive_weights = positive_weights
        self.constrain_pos = constrain_pos
        self.positive_spatial = positive_spatial
        self.factorize_spatial = factorize_spatial
        self.whitener = whitener
        self._regularizer_type = regularizer_type

        if self._regularizer_type == "adaptive_log_norm":
            self.gamma_sigma = gamma_sigma
            self.adaptive_neuron_reg_coefs = torch.nn.Parameter(
                torch.normal(mean=torch.ones(outdims, 1), std=torch.ones(outdims, 1))
            )
        elif self._regularizer_type != "l1":
            raise ValueError(f"regularizer_type should be 'l1' or 'adaptive_log_norm' but got {self._regularizer_type}")

        if positive_spatial and constrain_pos:
            warnings.warn(
                f"If both positive_spatial and constrain_pos are True, "
                f"only constrain_pos will effectively take place"
            )
        self.init_noise = init_noise
        self.normalize = normalize
        self.mean_activity = mean_activity
        self.spatial_reg_weight = spatial_reg_weight
        self.feature_reg_weight = self.resolve_deprecated_gamma_readout(
            feature_reg_weight, gamma_readout, default=1.0
        )
        self.temperature = temperature

        self._original_features = True
        self.initialize_features(**(shared_features or {}))

        if retinotopy_spatial is None and not fourier_spatial:
            self._retinotopy = False
            self._fourier = False
            if self.factorize_spatial:
                self.h_spatial = nn.Parameter(torch.Tensor(self.outdims, h, 1))
                self.w_spatial = nn.Parameter(torch.Tensor(self.outdims, 1, w))
            else:
                self.full_spatial = nn.Parameter(torch.Tensor(self.outdims, h, w))
        elif retinotopy_spatial is None and fourier_spatial:
            self._retinotopy = False
            self._fourier = True
            self.register_buffer("basis", real_fourier_basis(h, w, fourier_max_freq))
            k = self.basis.shape[0]
            self.coeffs = nn.Parameter(torch.Tensor(self.outdims, k))
            print('fourier coeffs shape:', self.coeffs.shape)
        elif retinotopy_spatial is not None and not fourier_spatial:
            self._retinotopy = True
            self._fourier = False
            self.retinotopy_spatial = RetinotopySpatial(source_grid, (self.outdims, h, w), self.factorize_spatial, **retinotopy_spatial)
        elif retinotopy_spatial is not None and fourier_spatial:
            self._retinotopy = True
            self._fourier = True
            self.retinotopy_spatial = FourierRetinotopy(source_grid, (self.outdims, h, w), fourier_max_freq, **retinotopy_spatial)

        if bias:
            bias = nn.Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.initialize()

    @property
    def spatial(self):
        if self._retinotopy:
            return self.retinotopy_spatial()
        else:
            if self._fourier:
                k = self.basis.shape[0]
                logits = torch.einsum("nk,khw->nhw", self.coeffs, self.basis) / math.sqrt(k)
                return logits
            elif self.factorize_spatial:
                h = self.h_spatial.shape[1]
                w = self.w_spatial.shape[2]
                spatial = (
                    self.h_spatial.expand(self.outdims, h, w) 
                    * self.w_spatial.expand(self.outdims, h, w)
                )
                return spatial
            else:
                return self.full_spatial

    @property
    def shared_features(self):
        return self._features

    @property
    def features(self):
        if self._shared_features:
            return self.scales * self._features[self.feature_sharing_index, ...]
        else:
            return self._features

    @property
    def weight(self):
        if self.positive_weights:
            self.features.data.clamp_min_(0)
        n = self.outdims
        c, h, w = self.in_shape
        return self.normalized_spatial.view(n, 1, w, h) * self.features.view(n, c, 1, 1)

    @property
    def normalized_spatial(self):
        """
        Normalize the spatial mask
        """
        weight = self.spatial
        if self.constrain_pos:
            weight.data.clamp_min_(0)
        elif self.positive_spatial:
            weight = torch.abs(weight)
        if self.normalize:
            # norm = weight.abs().sum(dim=1, keepdim=True)
            # norm = norm.sum(dim=2, keepdim=True).expand_as(self.spatial) + 1e-6
            # weight = weight / norm
            o, h, w = weight.shape
            weight = nn.functional.softmax(weight.view(o, h * w) / self.temperature, dim=1).view(o, h, w)
    
        return weight

    def adaptive_feature_l1_lognorm(self, reduction="sum", average=None):
        if self.whitener is not None:
            features = self.whitener.whiten_readouts(self.features.T).squeeze().T
        else:
            features = self.features
        features = self.adaptive_neuron_reg_coefs.abs() * features
        
        features_regularization = (
            self.apply_reduction(features.abs(), reduction=reduction, average=average) * self.feature_reg_weight
        )
        # adaptive_neuron_reg_coefs (betas) are supposted to be from lognorm distribution
        coef_prior = 1 / (self.gamma_sigma**2) * ((torch.log(self.adaptive_neuron_reg_coefs.abs()) ** 2).sum())
        return features_regularization + coef_prior

    def l1(self, reduction="sum", average=None):
        reduction = self.resolve_reduction_method(reduction=reduction, average=average)
        if reduction is None:
            raise ValueError("Reduction of None is not supported in this regularizer")

        n = self.outdims
        c, h, w = self.in_shape
        ret = (
            self.spatial.view(self.outdims, -1).abs().sum(dim=1, keepdim=True) * self.spatial_reg_weight
        ).sum()
        if reduction == "mean":
            ret = ret / (n * c * w * h)
        return ret

    def regularizer(self, reduction="sum", average=None):
        return (
            self.adaptive_feature_l1_lognorm(reduction=reduction, average=average)
            + self.l1(reduction=reduction, average=average)
        )

    def initialize(self, mean_activity=None):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """
        if mean_activity is None:
            mean_activity = self.mean_activity
        if not self._retinotopy:
            if self._fourier:
                self.coeffs.data.normal_(0, self.init_noise)
            elif self.factorize_spatial:
                self.h_spatial.data.normal_(0, self.init_noise)
                self.w_spatial.data.normal_(0, self.init_noise)
            else:
                self.full_spatial.data.normal_(0, self.init_noise)
        self._features.data.normal_(0, self.init_noise)
        if self._shared_features:
            self.scales.data.fill_(1.0)
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c = self.in_shape[0]
        if match_ids is not None:
            assert self.outdims == len(match_ids)

            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (
                    n_match_ids,
                    c,
                ), f"shared features need to have shape ({n_match_ids}, {c})"
                self._features = shared_features
                self._original_features = False
            else:
                self._features = nn.Parameter(
                    torch.Tensor(n_match_ids, c)
                )  # feature weights for each channel of the core
            self.scales = nn.Parameter(torch.Tensor(self.outdims, 1))  # feature weights for each channel of the core
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer("feature_sharing_index", torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = nn.Parameter(torch.Tensor(self.outdims, c))  # feature weights for each channel of the core
            self._shared_features = False

    def forward(self, x, shift=None, **kwargs):
        # if shift is not None:
        #     raise NotImplementedError("shift is not implemented for this readout")
        if self.constrain_pos:
            self.features.data.clamp_min_(0)

        c, h, w = x.size()[1:]
        c_in, h_in, w_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")

        if shift is not None:
            x = shift_feature_maps(x, shift)

        y_vec = torch.einsum("ncwh,owh->nco", x, self.normalized_spatial)
        if self.whitener is not None:
            _ = self.whitener(y_vec).squeeze()
            # if not self.training:
            #     y_vec = whitened
            
        y = torch.einsum("nco,oc->no", y_vec, self.features)
        if self.bias is not None:
            y = y + self.bias
        return y, y_vec

    def __repr__(self):
        c, h, w = self.in_shape
        r = self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"
        if self._shared_features:
            r += ", with {} features".format("original" if self._original_features else "shared")
        if self.normalize:
            r += ", normalized"
        else:
            r += ", unnormalized"
        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r


# Classes for backwards compatibility
class SpatialXFeatureLinear(FullFactorized2d):
    pass


class FullSXF(FullFactorized2d):
    pass
