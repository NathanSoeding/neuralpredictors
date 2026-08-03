import warnings
import math

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from .base import Readout


def peak_distance_penalty(m, radius=8.0, chunk=8192):
    """m: (n, h, w) softmax-normalized masks."""
    n, h, w = m.shape
    ar_h = torch.arange(h, device=m.device, dtype=m.dtype)
    ar_w = torch.arange(w, device=m.device, dtype=m.dtype)
    total = m.new_zeros(())
    for s in range(0, n, chunk):
        mc = m[s:s + chunk]
        b = mc.shape[0]
        idx = mc.reshape(b, -1).argmax(1)
        py = (idx // w).to(m.dtype)
        px = (idx % w).to(m.dtype)
        dy2 = (ar_h[None, :] - py[:, None]) ** 2          # (b, h)
        dx2 = (ar_w[None, :] - px[:, None]) ** 2          # (b, w)
        d = (dy2[:, :, None] + dx2[:, None, :]).sqrt()    # (b, h, w)
        total = total + (mc * F.relu(d - radius) ** 2).sum()
    return total / n


def concavity_penalty(b, W, lim, eps=1e-8):
    """b: (1,h,w) or (h,w) smoothed bias. W: (in_dim,h,w) smoothed ramps.
    lim: (in_dim,) = source_grid.abs().max(0).values."""
    b = b.reshape(b.shape[-2], b.shape[-1])
    signs = torch.cartesian_prod(*[torch.tensor([-1.0, 1.0], device=b.device)] * W.shape[0])
    signs = signs.reshape(-1, W.shape[0])                       # (2^d, d)
    rf = b[None] + torch.einsum("cd,dhw->chw", signs * lim[None], W)

    fxx = rf[:, 1:-1, 2:] - 2 * rf[:, 1:-1, 1:-1] + rf[:, 1:-1, :-2]
    fyy = rf[:, 2:, 1:-1] - 2 * rf[:, 1:-1, 1:-1] + rf[:, :-2, 1:-1]
    fxy = (rf[:, 2:, 2:] - rf[:, 2:, :-2] - rf[:, :-2, 2:] + rf[:, :-2, :-2]) / 4

    tr, diff = (fxx + fyy) / 2, (fxx - fyy) / 2
    lmax = tr + (diff ** 2 + fxy ** 2 + eps).sqrt()             # larger Hessian eigenvalue
    return F.relu(lmax).pow(2).mean()


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


class Factorized2d(Readout):
    def __init__(
        self,
        in_shape,  # channels, height, width
        outdims,
        bias,
        init_noise=1e-3,
        shared_features=None,
        mean_activity=None,
        feature_reg_weight=None,
        gamma_readout=None,
        regularizer_type="l1",
        gamma_sigma=0.1,
        source_grid=None,
        temperature=1.0,
        temp_per_neuron=False,
        kernel_size=7,
        kernel_sigma=2.0,
        smoothness_reg_weight=0.0,
        peak_distance_reg_weight=0.0,
        peak_distance_radius=8.0,
        concavity_reg_weight=0.0,
        **kwargs,
    ):
        
        super().__init__()

        h, w = in_shape[1:]  # channels, height, width
        self.in_shape = in_shape
        self.outdims = outdims

        self.temp_per_neuron = temp_per_neuron
        if temp_per_neuron:
            self.log_temp = nn.Parameter(torch.full((outdims, 1), math.log(temperature)))
        else:
            self.temperature = temperature
        
        self._regularizer_type = regularizer_type
        if self._regularizer_type == "adaptive_log_norm":
            self.gamma_sigma = gamma_sigma
            self.adaptive_neuron_reg_coefs = nn.Parameter(torch.randn(outdims, 1) + 1)
        elif self._regularizer_type != "l1":
            raise ValueError(f"regularizer_type should be 'l1' or 'adaptive_log_norm' but got {self._regularizer_type}")

        self.init_noise = init_noise
        self.mean_activity = mean_activity
        self.feature_reg_weight = self.resolve_deprecated_gamma_readout(
            feature_reg_weight, gamma_readout, default=1.0
        )
        self._original_features = True
        self.initialize_features(**(shared_features or {}))

        self.kernel_size = kernel_size
        self._smoothness_reg = smoothness_reg_weight > 0.0
        if self._smoothness_reg:
            self.smoothness_reg_weight = smoothness_reg_weight
            self.kernel_sigma = nn.Parameter(torch.tensor(float(kernel_sigma)))
        else:
            self.kernel_sigma = kernel_sigma

        self._peak_distance_reg = peak_distance_reg_weight > 0.0
        self.peak_distance_reg_weight = peak_distance_reg_weight
        self.peak_distance_radius = peak_distance_radius

        self._concavity_reg = concavity_reg_weight > 0.0
        self.concavity_reg_weight = concavity_reg_weight

        if source_grid is None:
            raise ValueError("factorized readout needs source grid for retinotopy mapping")
            
        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer("source_grid", torch.from_numpy(source_grid.astype(np.float32)))

        in_dim = source_grid.shape[1]
        self.spatial_w = nn.Parameter(torch.randn(in_dim, h, w))
        self.spatial_b = nn.Parameter(torch.zeros((1, h, w)))
    
        if bias:
            bias = nn.Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.initialize()

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
            self._features = nn.Parameter(torch.randn(self.outdims, c) * self.init_noise)  # feature weights for each channel of the core
            self._shared_features = False

    def initialize(self, mean_activity=None):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """
        if mean_activity is None:
            mean_activity = self.mean_activity
        if self._shared_features:
            self.scales.data.fill_(1.0)
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def feature_l1(self, whitener=None, reduction="sum", average=None):
        """
        Returns l1 regularization term for features.
        Args:
            average(bool): Deprecated (see reduction) if True, use mean of weights for regularization
            reduction(str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        """
        if self._original_features:
            if whitener:
                features = whitener.transform_weights(self.features.T).T
            else:
                features = self.features

            return self.apply_reduction(features.abs(), reduction=reduction, average=average)
        else:
            return 0

    def adaptive_feature_l1_lognorm(self, whitener=None, reduction="sum", average=None):
        if self._original_features:
            if whitener:
                features = whitener.transform_weights(self.features.T).T
            else:
                features = self.features
            
            features = self.adaptive_neuron_reg_coefs.abs() * features
            features_regularization = (
                self.apply_reduction(features.abs(), reduction=reduction, average=average) * self.feature_reg_weight
            )
            # adaptive_neuron_reg_coefs (betas) are supposted to be from lognorm distribution
            coef_prior = 1 / (self.gamma_sigma**2) * ((torch.log(self.adaptive_neuron_reg_coefs.abs()) ** 2).sum())
            return features_regularization + coef_prior
        else:
            return 0

    def exponential_smoothness(self):
        return torch.exp(-self.kernel_sigma) * self.smoothness_reg_weight

    def regularizer(self, whitener=None, reduction="sum", average=None):
        feature_reg = 0
        if self._regularizer_type == "l1":
            feature_reg = self.feature_l1(whitener=whitener, reduction=reduction, average=average) * self.feature_reg_weight
        elif self._regularizer_type == "adaptive_log_norm":
            feature_reg = self.adaptive_feature_l1_lognorm(whitener=whitener, reduction=reduction, average=average)
        else:
            raise NotImplementedError(f"Regularizer_type {self._regularizer_type} is not implemented")
        
        smoothness_reg = 0
        if self._smoothness_reg:
            smoothness_reg = self.exponential_smoothness()

        peak_distance_reg = 0
        if self._peak_distance_reg:
            peak_distance_reg = peak_distance_penalty(
                self.spatial, radius=self.peak_distance_radius
            ) * self.peak_distance_reg_weight

        concavity_reg = 0
        if self._concavity_reg:
            kernel = self.gaussian_kernel(self.spatial_b.device)
            b_smooth = self.smooth(self.spatial_b, kernel)
            W_smooth = self.smooth(self.spatial_w, kernel)
            lim = self.source_grid.abs().max(dim=0).values
            concavity_reg = concavity_penalty(b_smooth, W_smooth, lim) * self.concavity_reg_weight

        reg = feature_reg + smoothness_reg + peak_distance_reg + concavity_reg
        components = {
            'feature': feature_reg,
            'smoothness': smoothness_reg,
            'peak_distance': peak_distance_reg,
            'concavity': concavity_reg,
        }
        return reg, components

    def gaussian_kernel(self, device):
        x = torch.arange(self.kernel_size, device=device) - self.kernel_size // 2
        xx, yy = torch.meshgrid(x, x, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.kernel_sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def smooth(self, x, kernel):
        x = F.pad(x, (self.kernel_size // 2, ) * 4, mode='reflect')  # pad to preserve dimentions
        x = F.conv2d(x.unsqueeze(1), kernel).squeeze(1)
        return x
    
    @property
    def spatial(self):
        rf = torch.einsum('nd,dhw->nhw', self.source_grid, self.spatial_w)
        rf = rf + self.spatial_b
        
        kernel = self.gaussian_kernel(rf.device)
        rf = self.smooth(rf, kernel)
        
        n, h, w = rf.shape
        if self.temp_per_neuron:
            temp = self.log_temp.exp() + 1e-3
        else:
            temp = self.temperature
        normalized = F.softmax(rf.view(n, h * w) / temp, dim=1).view(n, h, w)
        return normalized

    @property
    def shared_features(self):
        return self._features

    @property
    def features(self):
        if self._shared_features:
            return self.scales * self._features[self.feature_sharing_index, ...]
        else:
            return self._features

    def forward(self, x, shift=None, **kwargs):
        c, h, w = x.size()[1:]
        c_in, h_in, w_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")

        if shift is not None:
            x = shift_feature_maps(x, shift)

        feature_vecs = torch.einsum("bchw,nhw->bnc", x, self.spatial)
            
        y = torch.einsum("bnc,nc->bn", feature_vecs, self.features)
        if self.bias is not None:
            y = y + self.bias
        return y, feature_vecs


class LegacyFullFactorized2d(Readout):
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
        spatial_and_feature_reg_weight=None,
        gamma_readout=None,
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
        if positive_spatial and constrain_pos:
            warnings.warn(
                f"If both positive_spatial and constrain_pos are True, "
                f"only constrain_pos will effectively take place"
            )
        self.init_noise = init_noise
        self.normalize = normalize
        self.mean_activity = mean_activity
        self.spatial_and_feature_reg_weight = self.resolve_deprecated_gamma_readout(
            spatial_and_feature_reg_weight, gamma_readout, default=1.0
        )

        self._original_features = True
        self.initialize_features(**(shared_features or {}))
        self.spatial = nn.Parameter(torch.Tensor(self.outdims, h, w))

        if bias:
            bias = nn.Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.initialize()

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
        if self.normalize:
            norm = self.spatial.pow(2).sum(dim=1, keepdim=True)
            norm = norm.sum(dim=2, keepdim=True).sqrt().expand_as(self.spatial) + 1e-6
            weight = self.spatial / norm
        else:
            weight = self.spatial
        if self.constrain_pos:
            weight.data.clamp_min_(0)
        elif self.positive_spatial:
            weight = torch.abs(weight)
        return weight

    def regularizer(self, whitener=None, reduction="sum", average=None):
        return self.l1(reduction=reduction, average=average) * self.spatial_and_feature_reg_weight

    def l1(self, reduction="sum", average=None):
        reduction = self.resolve_reduction_method(reduction=reduction, average=average)
        if reduction is None:
            raise ValueError("Reduction of None is not supported in this regularizer")

        n = self.outdims
        c, h, w = self.in_shape
        ret = (
            self.normalized_spatial.view(self.outdims, -1).abs().sum(dim=1, keepdim=True)
            * self.features.view(self.outdims, -1).abs().sum(dim=1)
        ).sum()
        if reduction == "mean":
            ret = ret / (n * c * w * h)
        return ret

    def initialize(self, mean_activity=None):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """
        if mean_activity is None:
            mean_activity = self.mean_activity
        self.spatial.data.normal_(0, self.init_noise)
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
        if shift is not None:
            raise NotImplementedError("shift is not implemented for this readout")
        if self.constrain_pos:
            self.features.data.clamp_min_(0)

        c, h, w = x.size()[1:]
        c_in, h_in, w_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError("the specified feature map dimension is not the readout's expected input dimension")

        feature_vecs = torch.einsum("ncwh,owh->nco", x, self.normalized_spatial)
        y = torch.einsum("nco,oc->no", feature_vecs, self.features)
        if self.bias is not None:
            y = y + self.bias
        return y, feature_vecs

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
class SpatialXFeatureLinear(LegacyFullFactorized2d):
    pass


class FullSXF(LegacyFullFactorized2d):
    pass
