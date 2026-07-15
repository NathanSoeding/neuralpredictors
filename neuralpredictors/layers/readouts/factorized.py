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

def real_fourier_basis(H, W, max_freq, diagonal):
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
            if u == 0 and v == 0:
                continue
            if (not diagonal) and u != 0 and v != 0:
                continue

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
    def __init__(self, source_grid, out_shape, max_freq=3, diagonal=True, hidden_features=20, hidden_layers=0, in_dim=None):
        super().__init__()

        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer("source_grid", torch.from_numpy(source_grid.astype(np.float32)))
        self.out_shape = out_shape

        n, h, w = self.out_shape
        self.register_buffer("basis", real_fourier_basis(h, w, max_freq, diagonal))
        k = self.basis.shape[0]

        in_dim = source_grid.shape[1] if in_dim is None else in_dim
        self.inp_dim = in_dim
        layers = []
        for hidden_dim in [hidden_features] * hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, k))
        self.mlp = nn.Sequential(*layers)

    def forward(self, pupil_center=None):
        k = self.basis.shape[0]
        if pupil_center is not None:
            n, _ = self.source_grid.shape
            combined = torch.cat([
                self.source_grid, 
                pupil_center.unsqueeze(0).expand(n, -1),
            ], dim=-1)    
            coeffs = self.mlp(combined)  # n, k
        else:
            coeffs = self.mlp(self.source_grid)  # n, k
        
        logits = torch.einsum("nk,khw->nhw", coeffs, self.basis) / math.sqrt(k)
        return logits

class GaussianRetinotopy(nn.Module):
    def __init__(self, source_grid, out_shape, predict_sigma=False, init_sigma=0.05, sigma_range=(0.01, 0.3), hidden_features=20, hidden_layers=0, in_dim=None):
        super().__init__()
        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer("source_grid", torch.from_numpy(source_grid.astype(np.float32)))

        self.out_shape = out_shape
        n, h, w = self.out_shape
        self.predict_sigma = predict_sigma
        self.sigma_range = sigma_range

        in_dim = source_grid.shape[1] if in_dim is None else in_dim
        self.inp_dim = in_dim

        # output dim: 2 for (x, y) center, +1 more if predicting sigma
        out_dim = 3 if predict_sigma else 2

        layers = []
        for hidden_dim in [hidden_features] * hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

        if not predict_sigma:
            # fixed, learnable scalar sigma (in normalized [-1,1] coord units)
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(init_sigma))))

        # precompute pixel grid once (buffers, not parameters)
        ys = torch.linspace(-1, 1, h) * (h / max(h, w))
        xs = torch.linspace(-1, 1, w) * (w / max(h, w))
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (h, w)
        self.register_buffer("grid_x", grid_x.clone())
        self.register_buffer("grid_y", grid_y.clone())

    def forward(self, pupil_center=None):
        n, _ = self.source_grid.shape

        if pupil_center is not None:
            combined = torch.cat([
                self.source_grid,
                pupil_center.unsqueeze(0).expand(n, -1),
            ], dim=-1)
            out = self.mlp(combined)  # n, out_dim
        else:
            out = self.mlp(self.source_grid)  # n, out_dim

        # center coords, squashed to [-1, 1]
        center = torch.tanh(out[:, :2])  # n, 2  (x, y)

        h, w = self.out_shape[1], self.out_shape[2]
        scale = torch.tensor([w, h], dtype=center.dtype, device=center.device) / max(h, w)
        center = center * scale  # now in same coordinate space as grid_x/grid_y

        if self.predict_sigma:
            lo, hi = self.sigma_range
            sigma = lo + (hi - lo) * torch.sigmoid(out[:, 2])  # n,
        else:
            sigma = self.log_sigma.exp().expand(n)  # n,

        px = center[:, 0].view(n, 1, 1)  # (n,1,1)
        py = center[:, 1].view(n, 1, 1)
        sigma = sigma.view(n, 1, 1)

        grid_x = self.grid_x.unsqueeze(0)  # (1,h,w)
        grid_y = self.grid_y.unsqueeze(0)

        dist_sq = (grid_x - px) ** 2 + (grid_y - py) ** 2
        logits = -dist_sq / (2 * sigma ** 2)  # log-gaussian, unnormalized (peak at 0)

        return logits

class DiscretizedRetinotopy(nn.Module):
    def __init__(self, source_grid, out_shape, kernel_size=7, sigma=2.0, in_dim=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer("source_grid", torch.from_numpy(source_grid.astype(np.float32)))

        n, h, w = out_shape
        in_dim = source_grid.shape[1] if in_dim is None else in_dim
        self.inp_dim = in_dim

        self.weights = torch.nn.Parameter(
            torch.normal(mean=torch.ones(in_dim, 1, h, w), std=torch.ones(in_dim, 1, h, w))
        )
        self.bias = torch.nn.Parameter(
            torch.normal(mean=torch.ones(1, 1, h, w), std=torch.ones(1, 1, h, w))
        )

    @property
    def gaussian_kernel(self):
        x = torch.arange(self.kernel_size) - self.kernel_size // 2
        xx, yy = torch.meshgrid(x, x, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        kernel /= kernel.sum()
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

    @property
    def smooth_w(self):
        d, _, h, w = self.weights.shape
        kernel = self.gaussian_kernel.to(self.source_grid.device)
        weight = F.conv2d(self.weights, kernel, padding='same').view(d, h, w)
        return weight

    @property
    def smooth_b(self):
        _, _, h, w = self.bias.shape
        kernel = self.gaussian_kernel.to(self.source_grid.device)
        bias = F.conv2d(self.bias, kernel, padding='same').view(1, h, w)
        return bias

    def forward(self, pupil_center=None):
        n, _ = self.source_grid.shape

        if pupil_center is not None:
            combined = torch.cat([
                self.source_grid,
                pupil_center.unsqueeze(0).expand(n, -1),
            ], dim=-1)  # n, in_dim
        else:
            combined = self.source_grid  # n, in_dim

        x = torch.einsum('nd,dhw->nhw', combined, self.smooth_w)
        x = x + self.smooth_b
        return x

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
        factorize_spatial=False,
        regularizer_type="l1",
        gamma_sigma=0.1,
        source_grid=None,
        retinotopy_spatial=None,
        fourier_spatial=False,
        fourier_max_freq=4,
        whitener=None,
        init_temp=1.0,
        hard=False,
        normalize_logits=False,
        retinotopy_in_dim=2,
        entropy_reg=False,
        entropy_reg_weight=1.0,
        diagonal=True,
        com_reg_weight=0.0,
        gaussian_spatial=False,
        predict_sigma=False,
        init_sigma=1.0,
        discretized_spatial=False,
        kernel_size=7,
        sigma=2.0,
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
        self.temperature = init_temp
        self._regularizer_type = regularizer_type
        self.hard = hard
        self.normalize_logits = normalize_logits
        self.entropy_reg = entropy_reg
        self.entropy_reg_weight = entropy_reg_weight
        self.com_reg_weight = com_reg_weight

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
            self.register_buffer("basis", real_fourier_basis(h, w, fourier_max_freq, diagonal))
            k = self.basis.shape[0]
            self.coeffs = nn.Parameter(torch.Tensor(self.outdims, k))
            print('fourier coeffs shape:', self.coeffs.shape)
        elif retinotopy_spatial is not None and fourier_spatial:
            self._retinotopy = True
            self._fourier = True
            self.retinotopy_spatial = FourierRetinotopy(source_grid, (self.outdims, h, w), fourier_max_freq, diagonal, **retinotopy_spatial)
        elif retinotopy_spatial is not None and gaussian_spatial:
            self._retinotopy = True
            self._gaussian = True
            self.retinotopy_spatial = GaussianRetinotopy(source_grid, (self.outdims, h, w), predict_sigma, init_sigma, **retinotopy_spatial)
        elif retinotopy_spatial is not None and discretized_spatial:
            self._retinotopy = True
            self._discretized = True
            self.retinotopy_spatial = DiscretizedRetinotopy(source_grid, (self.outdims, h, w), kernel_size, sigma, retinotopy_spatial['in_dim'])
        elif retinotopy_spatial is not None:
            self._retinotopy = True
            self._fourier = False
            self.retinotopy_spatial = RetinotopySpatial(source_grid, (self.outdims, h, w), self.factorize_spatial, **retinotopy_spatial)

        if bias:
            bias = nn.Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.initialize()

    def spatial(self, pupil_center=None):
        if self._retinotopy:
            return self.retinotopy_spatial(pupil_center)
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

    # @property
    # def weight(self):
    #     if self.positive_weights:
    #         self.features.data.clamp_min_(0)
    #     n = self.outdims
    #     c, h, w = self.in_shape
    #     return self.normalized_spatial.view(n, 1, w, h) * self.features.view(n, c, 1, 1)

    def normalized_spatial(self, pupil_center=None):
        """
        Normalize the spatial mask
        """
        weight = self.spatial(pupil_center=pupil_center)
        o, h, w = weight.shape
        if self.constrain_pos:
            weight.data.clamp_min_(0)
        elif self.positive_spatial:
            weight = torch.abs(weight)
        if self.normalize:
            # norm = weight.abs().sum(dim=1, keepdim=True)
            # norm = norm.sum(dim=2, keepdim=True).expand_as(self.spatial) + 1e-6
            # weight = weight / norm
            
            if self.normalize_logits:
                z = weight.view(o, h * w)
                z = z - z.mean(dim=-1, keepdim=True)
                z = z / (z.std(dim=-1, keepdim=True) + 1e-5)
                weight = F.softmax(z / self.temperature, dim=-1).view(o, h, w)
            else:
                weight = F.softmax(weight.view(o, h * w) / self.temperature, dim=1).view(o, h, w)
            
            # straight through trick to get onehot outputs while preserving gradients
            if self.hard:
                flat = weight.view(o, h * w)
                argmax = flat.argmax(-1)
                onehot = torch.zeros_like(flat)
                onehot.scatter_(1, argmax[:, None], 1.0)
                onehot = onehot.view(o, h, w)
                weight = onehot.detach() - weight.detach() + weight
    
        return weight

    def adaptive_feature_l1_lognorm(self, features, reduction="sum", average=None):
        scaled = self.adaptive_neuron_reg_coefs.abs() * features
        
        features_regularization = (
            self.apply_reduction(scaled.abs(), reduction=reduction, average=average) * self.feature_reg_weight
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
    
    def entropy_regularizer(self, features):
        n, c = features.shape  # n x c
        normalized = features.abs().sum(0) / (features.abs().sum() * n)
        penalty = -(normalized * normalized.log()).sum() * self.entropy_reg_weight

        return penalty

    def center_of_mass(self):
        """ Penalize neurons whose RF center deviates from the image center. """
        spatial = self.normalized_spatial()

        o, h, w = spatial.shape

        y = torch.arange(h, device=spatial.device, dtype=spatial.dtype)
        x = torch.arange(w, device=spatial.device, dtype=spatial.dtype)

        y_grid = y.view(1, h, 1)
        x_grid = x.view(1, 1, w)

        mass = spatial.abs()
        normalized_mass = mass / mass.sum(dim=(1, 2), keepdim=True) + 1e-6

        x_cm = (normalized_mass * x_grid).sum(dim=(1, 2))
        y_cm = (normalized_mass * y_grid).sum(dim=(1, 2))

        x_center = (w - 1) / 2.0
        y_center = (h - 1) / 2.0

        penalty = ((x_cm - x_center).pow(2) + (y_cm - y_center).pow(2)).sqrt()
        
        return penalty.sum()

    def regularizer(self, reduction="sum", average=None):
        if self.whitener is not None:
            features = self.whitener.whiten_readouts(self.features.T).squeeze().T
        else:
            features = self.features

        penalty = 0
        if self.feature_reg_weight > 0:
            penalty += self.adaptive_feature_l1_lognorm(features, reduction=reduction, average=average)
        if self.entropy_reg:
            penalty += self.entropy_regularizer(features)
        if self.com_reg_weight > 0:
            penalty += self.center_of_mass() * self.com_reg_weight
        return penalty

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

    def forward(self, x, shift=None, temp=None, pupil_center=None, **kwargs):
        if temp is not None:
            self.temperature = temp  # for temperature annealing

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

        if pupil_center is None:
            y_vec = torch.einsum("ncwh,owh->nco", x, self.normalized_spatial())
        else:
            # loop because of memory
            y_vec = torch.stack([
                torch.einsum("cwh,owh->co", img, self.normalized_spatial(pupil_center=pupil))
                for img, pupil in zip(x, pupil_center)
            ])
                
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
