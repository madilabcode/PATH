import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AUCMarginLoss(nn.Module):
    """
    AUC margin loss from:
    'Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and
     Empirical Studies on Medical Image Classification' (Yuan et al., 2021).
    
    This implements the batch Monte-Carlo estimate of Eq. (6):

        AM(w) ≈ Var_pos + Var_neg + (m + mean_neg - mean_pos)_+^2

    Args:
        margin (float): margin m in the paper (default: 1.0).

    Expected input:
        y_pred: shape (N,) or (N, 1), real-valued scores.
        y_true: shape (N,), labels in {0,1} or {-1,1}.
    """
    def __init__(self, margin: float = 1.0, reduction='None'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        if y_true.min() < 0:
            y_true01 = (y_true > 0).float()
        else:
            y_true01 = y_true.float()

        pos_mask = (y_true01 == 1)
        neg_mask = ~pos_mask

        if not pos_mask.any() or not neg_mask.any():
            return torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)

        pos_scores = y_pred[pos_mask]
        neg_scores = y_pred[neg_mask]

        mean_pos = pos_scores.mean()
        mean_neg = neg_scores.mean()

        var_pos = ((pos_scores - mean_pos) ** 2).mean()
        var_neg = ((neg_scores - mean_neg) ** 2).mean()

        margin_term = F.relu(self.margin + mean_neg - mean_pos) ** 2

        loss = var_pos + var_neg + margin_term
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'None':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
