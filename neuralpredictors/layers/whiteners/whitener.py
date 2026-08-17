from collections import deque
from contextlib import contextmanager

import torch
import torch.nn as nn


class Whitener(nn.Module):
    def __init__(self, model_dim, momentum=0.003, mode='ema', detach_batch_stats=True, window_size=1, eps=1e-5):
        super().__init__()

        assert mode in ('ema', 'batch'), f"mode must be 'ema' or 'batch', got {mode!r}"
        self.mode = mode
        self.detach_batch_stats = detach_batch_stats
        self.window_size = window_size
        self._history = deque(maxlen=max(window_size - 1, 0))
        self.eps = eps

        self.register_buffer('mu_ema', torch.zeros(1, model_dim))
        self.register_buffer('cov_ema', torch.eye(model_dim))
        self.momentum = momentum

        self.last_min = None
        self.last_max = None
        self.last_cond = None
        self.last_min_abs_eigenvalue = None
        self.last_max_abs_eigenvalue = None
        self.last_raw_cond = None
        self.last_raw_min_abs_eigenvalue = None
        self.last_raw_max_abs_eigenvalue = None

        self._live_stats_mode = False

    @contextmanager
    def live_stats(self):
        """
        Diagnostic-only context: while active, forward() whitens each batch using that
        batch's own statistics alone (no pooling with other batches), and touches no
        persistent state (mu_ema/cov_ema/_history/diagnostics are left untouched). Use
        this to evaluate with live single-batch statistics without contaminating the
        real EMA/window state used for training and normal inference.
        """
        self._live_stats_mode = True
        try:
            yield
        finally:
            self._live_stats_mode = False

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

    def _pooled_stats(self, X):
        n_cur = X.shape[0]
        sum_cur = X.sum(dim=0, keepdim=True)
        sumsq_cur = X.T @ X

        n = n_cur + sum(n_i for n_i, _, _ in self._history)
        s1 = sum_cur + sum((s_i for _, s_i, _ in self._history), torch.zeros_like(sum_cur))
        s2 = sumsq_cur + sum((q_i for _, _, q_i in self._history), torch.zeros_like(sumsq_cur))

        self._history.append((n_cur, sum_cur.detach(), sumsq_cur.detach()))

        mu = s1 / n
        cov = (s2 - n * (mu.transpose(0, 1) @ mu)) / (n - 1)
        return mu, cov

    def _whiten_with(self, x, mu, cov):
        eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        L = torch.linalg.cholesky(cov + self.eps * eye)
        centered = x - mu.view(1, 1, x.shape[-1])  # (B, N, D)
        whitened = torch.linalg.solve_triangular(L, centered.transpose(-1, -2), upper=False).transpose(-1, -2)
        return whitened

    @staticmethod
    def _cond_and_extreme_abs_eigenvalues(cov):
        # cov is symmetric PSD, so its singular values equal |eigenvalues|; one svdvals()
        # call gives the condition number and both extreme-magnitude eigenvalues for the
        # cost of a single decomposition.
        singular_values = torch.linalg.svdvals(cov)
        max_sv = singular_values[0].item()
        min_sv = singular_values[-1].item()
        return max_sv / min_sv, min_sv, max_sv

    def _update_diagnostics(self, whitened, cov):
        with torch.no_grad():
            self.last_min = whitened.min().item()
            self.last_max = whitened.max().item()

            Wc = whitened.detach().flatten(0, 1)
            Wc = Wc - Wc.mean(dim=0, keepdim=True)
            whitened_cov = Wc.T @ Wc / (Wc.shape[0] - 1)

            self.last_cond, self.last_min_abs_eigenvalue, self.last_max_abs_eigenvalue = (
                self._cond_and_extreme_abs_eigenvalues(whitened_cov)
            )
            self.last_raw_cond, self.last_raw_min_abs_eigenvalue, self.last_raw_max_abs_eigenvalue = (
                self._cond_and_extreme_abs_eigenvalues(cov.detach())
            )

    def whiten(self, x):
        return self._whiten_with(x, self.mu_ema, self.cov_ema)

    def transform_weights(self, w):
        eye = torch.eye(self.cov_ema.shape[0], dtype=self.cov_ema.dtype, device=self.cov_ema.device)
        L = torch.linalg.cholesky(self.cov_ema + self.eps * eye)
        return L.T @ w

    def forward(self, x):
        assert self.mode == 'batch', "forward() only applies batch-level whitening; use it only when mode='batch'"

        if self._live_stats_mode:
            X = x.detach().flatten(0, 1)
            mu = X.mean(dim=0, keepdim=True)
            Xc = X - mu
            cov = Xc.T @ Xc / (X.shape[0] - 1)
            return self._whiten_with(x, mu, cov)

        if self.training:
            self.update(x)

            X = x.flatten(0, 1)
            if self.detach_batch_stats:
                X = X.detach()
            mu, cov = self._pooled_stats(X)
        else:
            mu, cov = self.mu_ema, self.cov_ema

        whitened = self._whiten_with(x, mu, cov)
        self._update_diagnostics(whitened, cov)
        return whitened
