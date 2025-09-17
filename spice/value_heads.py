from __future__ import annotations
from typing import Tuple, List
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrozenPrior(nn.Module):
    """
    Small MLP prior; weights are frozen. Distinct per head (random init).
    Input is [h_s ; one_hot(a)].
    """
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        for p in self.net.parameters():
            p.requires_grad = False  # never trained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.net(x)  # [B,1]


class QHead(nn.Module):
    """
    Single Q head: Q_k([h_s; a]) = MLP([h_s; a]) + α * p_k([h_s; a]),
    where p_k is a frozen prior function (random features).
    """
    def __init__(self, in_dim: int, hidden: int = 256, alpha_prior: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.prior = FrozenPrior(in_dim, hidden=hidden)
        self.alpha = alpha_prior

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x) + self.alpha * self.prior(x)  # [B,1]


class EnsembleQ(nn.Module):
    """
    K-head ensemble. Input is concatenated [h_s; one_hot(a)].
    forward(x) returns [B, K].
    Also supports an optional 'anchor loss' to keep heads near their own initial params
    (helps preserve diversity without collapse).
    """
    def __init__(self, K: int, in_dim: int, hidden: int = 256, alpha_prior: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([QHead(in_dim, hidden, alpha_prior) for _ in range(K)])

        # capture initial parameters as anchors (not registered as buffers to keep ckpts light)
        self._anchors: List[List[torch.Tensor]] = []
        self._capture_anchors()

    def _capture_anchors(self):
        self._anchors = []
        for h in self.heads:
            self._anchors.append([p.detach().clone() for p in h.parameters()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # concat heads along last dim -> [B, K]
        return torch.cat([h(x) for h in self.heads], dim=-1)

    @staticmethod
    def mean_and_std(qs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = qs.mean(dim=-1, keepdim=True)
        std  = qs.std(dim=-1, keepdim=True)
        return mean, std

    def anchor_loss(self) -> torch.Tensor:
        """
        Sum_k ||θ_k - θ_k^0||^2 over all heads/params. Cheap (tiny λ recommended).
        """
        if not self._anchors:
            # No anchors captured (should not happen); return a zero scalar
            return torch.zeros((), device=next(self.parameters()).device)

        loss = torch.zeros((), device=next(self.parameters()).device)
        for h, anchor_params in zip(self.heads, self._anchors):
            for p, a in zip(h.parameters(), anchor_params):
                loss = loss + torch.sum((p - a.to(p.device))**2)
        return loss


def make_ensemble_q(K: int, d_model: int, action_dim: int,
                    hidden: int = 256, alpha_prior: float = 0.1) -> EnsembleQ:
    """
    Build an ensemble that expects inputs shaped like [h_s ; one_hot(a)].
    In-dim is d_model + action_dim to match Transformer embedding + one-hot action.
    """
    in_dim = d_model + action_dim
    return EnsembleQ(K=K, in_dim=in_dim, hidden=hidden, alpha_prior=alpha_prior)
