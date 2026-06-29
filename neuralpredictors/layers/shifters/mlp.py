import logging

import torch
from torch import nn
from torch.nn import ModuleDict
from torch.nn.init import xavier_normal_
import math

from .base import Shifter

logger = logging.getLogger(__name__)


class MLP(Shifter):
    def __init__(self, input_features=2, hidden_channels=10, shift_layers=1, bias=True,
                 stochastic=False, init_noise=1.0, learn_covariance=True, **kwargs):
        """
        Multi-layer perceptron shifter
        Args:
            input_features (int): number of input features, defaults to 2.
            hidden_channels (int): number of hidden units.
            shift_layers (int): number of shifter layers (n=1 will correspond to a network without a hidden layer).
            bias (bool): whether to use bias in linear layers.
            stochastic (bool): if True, samples from a Gaussian during training via the reparameterization trick.
            init_noise (float): initial noise strength (std). Only used when stochastic=True.
            learn_covariance (bool): if True, covariance is learned via a Cholesky parameterization.
                                     If False, uses a fixed scaled identity init_noise^2 * I.
                                     Only used when stochastic=True.
            **kwargs:
        """
        super().__init__()
        prev_output = input_features
        feat = []
        for _ in range(shift_layers - 1):
            feat.extend([nn.Linear(prev_output, hidden_channels, bias=bias), nn.Tanh()])
            prev_output = hidden_channels
        feat.extend([nn.Linear(prev_output, 2, bias=bias), nn.Tanh()])
        self.mlp = nn.Sequential(*feat)

        self.stochastic = stochastic
        self.init_noise = init_noise
        self.learn_covariance = learn_covariance

        if self.stochastic and self.learn_covariance:
            # Learned covariance via Cholesky factor L (lower triangular), so Σ = L @ L.T
            # log-diagonal ensures positive diagonal entries after exp().
            self.chol_log_diag = nn.Parameter(torch.zeros(2))  # log of diagonal: [log L_00, log L_11]
            self.chol_off_diag = nn.Parameter(torch.zeros(1))  # lower-triangular off-diagonal: [L_10]

        self.initialize()

    def _cholesky_factor(self, device, dtype):
        """Builds the 2x2 lower-triangular Cholesky factor L with positive diagonal.
        If learn_covariance=False, returns a fixed scaled identity: init_noise * I."""
        if self.learn_covariance:
            L = torch.zeros(2, 2, device=self.chol_log_diag.device, dtype=self.chol_log_diag.dtype)
            L[0, 0] = torch.exp(self.chol_log_diag[0])
            L[1, 0] = self.chol_off_diag[0]
            L[1, 1] = torch.exp(self.chol_log_diag[1])
        else:
            L = torch.eye(2, device=device, dtype=dtype) * self.init_noise
        return L  # Σ = L @ L.T

    def covariance(self):
        """Returns the full 2x2 covariance matrix Σ = L @ L.T."""
        L = self._cholesky_factor()
        return L @ L.T

    def regularizer(self):
        return 0

    def initialize(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        if self.stochastic and self.learn_covariance:
            nn.init.constant_(self.chol_log_diag, math.log(self.init_noise))
            nn.init.zeros_(self.chol_off_diag)

    def forward(self, pupil_center, trial_idx=None):
        if trial_idx is not None:
            pupil_center = torch.cat((pupil_center, trial_idx), dim=1)
        if not self.mlp[0].in_features == pupil_center.shape[1]:
            raise ValueError(
                "The expected input shape of the shifter and the shape of the input do not match! "
                "(Maybe due to the appending of trial_idx to pupil_center?)"
            )

        mu = self.mlp(pupil_center)

        if self.stochastic and self.training:
            # Reparameterization trick: z = mu + L @ eps,  eps ~ N(0, I)
            L = self._cholesky_factor(device=mu.device, dtype=mu.dtype)   # (2, 2)
            eps = torch.randn_like(mu)    # (batch, 2)
            return mu + (L @ eps.T).T

        return mu


class MLPShifter(ModuleDict):
    def __init__(
        self,
        data_keys,
        input_channels=2,
        hidden_channels_shifter=2,
        shift_layers=1,
        gamma_shifter=0,
        bias=True,
        stochastic=False,
        init_noise=1.0,
        learn_covariance=True,
        **kwargs
    ):
        """
        Args:
            data_keys (list of str): keys of the shifter dictionary, correspond to the data_keys of the nnfabirk dataloaders
            gamma_shifter: weight of the regularizer

            See docstring of base class for the other arguments.
        """
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(k, MLP(input_channels, hidden_channels_shifter, shift_layers, bias, stochastic, init_noise, learn_covariance))

    def initialize(self, **kwargs):
        pass

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_shifter
