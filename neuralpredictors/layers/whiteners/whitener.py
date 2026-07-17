import torch
import torch.nn as nn


class Whitener(nn.Module):
    def __init__(self, model_dim, momentum=0.003):
        super().__init__()

        self.register_buffer('mu_ema', torch.zeros(1, model_dim))
        self.register_buffer('cov_ema', torch.eye(model_dim))
        self.momentum = momentum

    def update(self, features):
        batch, neurons, c = features.shape
        
        X = features.detach().flatten(0, 1)
        mu_hat = X.mean(dim=0, keepdims=True)

        self.mu_ema.copy_(
            (1 - self.momentum) * self.mu_ema
            + self.momentum * mu_hat
        )
        
        Xc = X - self.mu_ema

        sig_hat = Xc.T @ Xc / (batch * neurons - 1)

        self.cov_ema.copy_(
            (1 - self.momentum) * self.cov_ema 
            + self.momentum * sig_hat
        )
        
    def whiten(self, x, eps=1e-5):
        eye = torch.eye(self.cov_ema.shape[0], dtype=self.cov_ema.dtype, device=self.cov_ema.device)
        L = torch.linalg.cholesky(self.cov_ema + eps * eye)
        centered = x - self.mu_ema.view(1, 1, x.shape[-1])  # (B, N, D)
        whitened = torch.linalg.solve_triangular(L, centered.transpose(-1, -2), upper=False).transpose(-1, -2)
        return whitened
    
    def transform_weights(self, w, eps=1e-5):
        eye = torch.eye(self.cov_ema.shape[0], dtype=self.cov_ema.dtype, device=self.cov_ema.device)
        L = torch.linalg.cholesky(self.cov_ema + eps * eye)
        return L.T @ w
