import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





class SoftCLIPCompact(nn.Module):
    """
    Minimal soft-CLIP (cross-modal only) with Top-K target focusing.

    Options:
      - Provide a symmetric similarity A [N,N] to build targets, OR pass P_ab/P_ba directly.
      - target_mode: 'power' (ReLU(A)^gamma then row-normalize) or 'softmax' (row-softmax(A/tau_P)).
      - topk: keep only top-k targets per row before normalizing (focus gradients).

    Args:
      temperature (float): logits temperature for cross-modal softmax.
      target_mode (str): 'power' | 'softmax' for building targets from A.
      gamma (float): sharpening for 'power' mode.
      tau_P (float): temperature for 'softmax' mode (targets).
      topk (int|None): Top-K for targets (None disables).
      reduction (str): 'mean' | 'sum' | 'none'.
    """

    def __init__(self, temperature=0.5, target_mode='softmax',
                 gamma=3, tau_P=1.0, topk=16, reduction='mean'):
        super().__init__()
        self.temperature = float(temperature)
        self.target_mode = target_mode
        self.gamma = float(gamma)
        self.tau_P = float(tau_P)
        self.topk = topk
        self.reduction = reduction

    @staticmethod
    def _row_normalize(W, eps=1e-8):
        W = W.clamp_min(0)
        Z = W.sum(dim=1, keepdim=True)
        eye = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        return torch.where(Z > eps, W / (Z + eps), eye)

    @staticmethod
    def _keep_topk_rows(X, k):
        if k is None or k >= X.size(1):
            return X
        vals, idx = torch.topk(X, k=k, dim=1)
        keep = torch.zeros_like(X, dtype=torch.bool)
        keep.scatter_(1, idx, True)
        return torch.where(keep, X, torch.zeros_like(X))

    def _build_targets_from_A(self, A):
        """
        Build P_ab (and P_ba) from a symmetric similarity A [N,N].
        Diagonal is zeroed so self doesn't dominate.
        """
        A = A.clone()
        A.fill_diagonal_(0.0)

        if self.target_mode == 'softmax':
            P_ab = F.softmax(A / self.tau_P, dim=1)
        elif self.target_mode == 'power':
            W = A.clamp_min(0.0)
            if self.gamma != 1.0:
                W = W.pow(self.gamma)
            if self.topk is not None:
                W = self._keep_topk_rows(W, self.topk)
            P_ab = W#self._row_normalize(W)
        else:
            raise ValueError("target_mode must be 'power' or 'softmax'")

        # symmetric A ⇒ same targets both directions
        P_ba = P_ab
        return P_ab, P_ba

    def _soft_ce(self, logits, P):
        # -sum_j P_ij * log_softmax(logits_i,:)
        return (-(P * (logits - torch.logsumexp(logits, dim=1, keepdim=True))).sum(dim=1))
    
    def _info_nce_style(self, logits, P):
        """
        InfoNCE-style loss with explicit numerator and denominator
        """
        # Numerator: weighted positive similarities
        numerator = torch.exp(logits) * P  # [N,N]
        numerator_sum = numerator.sum(dim=1)  # [N] - sum of weighted positives
        
        # Denominator: all similarities
        denominator = torch.exp(logits).sum(dim=1)  # [N] - sum of all similarities
        
        # InfoNCE-like loss
        loss = -torch.log(numerator_sum / (denominator + 1e-10))
        return loss

    def forward(self, a, b, A=None, P_ab=None, P_ba=None, l2_normalize=True):
        """
        a, b: [N,D] embeddings (paired in order).
        A: optional [N,N] symmetric similarity to build soft targets from.
        P_ab, P_ba: optional prebuilt row-stochastic targets (override A if provided).
        """
        if l2_normalize:
            a = F.normalize(a, dim=-1)
            b = F.normalize(b, dim=-1)

        N, dev = a.size(0), a.device
        tau = self.temperature

        if P_ab is None or P_ba is None:
            if A is None:
                # fall back to hard CLIP if nothing given
                P_ab = torch.eye(N, device=dev, dtype=a.dtype)
                P_ba = P_ab
                print("Using hard CLIP")
            else:
                P_ab, P_ba = self._build_targets_from_A(A)

        # Cross-modal logits
        L_ab = (a @ b.T) / tau
        L_ba = L_ab.T

        # InfoNCE-style loss both ways with explicit numerator/denominator
        loss = 0.5 * (self._info_nce_style(L_ab, P_ab) + self._info_nce_style(L_ba, P_ba))  # [N]

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        raise ValueError(f"Invalid reduction: {self.reduction}")


class SoftCLIPWithIntraModal(SoftCLIPCompact):
    """
    SoftCLIP with intra-modal components using one similarity matrix A.
    Inherits from SoftCLIPCompact and adds intra-modal losses.
    
    Args:
        temperature (float): logits temperature
        target_mode (str): 'power' | 'softmax' for building targets from A
        gamma (float): sharpening for 'power' mode
        tau_P (float): temperature for 'softmax' mode (targets)
        topk (int|None): Top-K for targets (None disables)
        cross_weight (float): weight for cross-modal loss (a-b)
        intra_weight (float): weight for intra-modal losses (a-a, b-b)
        reduction (str): 'mean' | 'sum' | 'none'
    """
    
    def __init__(self, temperature=0.5, target_mode='softmax', gamma=1.5, tau_P=1.0, topk=16,
                 cross_weight=1.0, intra_weight=0.5, reduction='mean', hard_clip=False):
        super().__init__(temperature, target_mode, gamma, tau_P, topk, reduction)
        self.cross_weight = cross_weight
        self.intra_weight = intra_weight
        self.hard_clip = hard_clip

    def _info_nce_style(self, logits_x2v, logits_v2v, P):
        """
        InfoNCE-style loss with explicit numerator and denominator
        """
        # Numerator: weighted positive similarities
        numerator = torch.exp(logits_x2v) * P  # [N,N]
        numerator_sum = numerator.sum(dim=1)  # [N] - sum of weighted positives
        # Denominator: all similarities
        denominator = (torch.exp(logits_x2v)).sum(dim=1)  # [N] - sum of all similarities
        denominator += (torch.exp(logits_v2v) * (1-P)).sum(dim=1)  # [N] - sum of all similarities
        
        # InfoNCE-like loss
        loss = -torch.log((numerator_sum + 1e-10) / (denominator))
        return loss

    def forward(self, a, b, A=None, l2_normalize=True, slides=None):
        """
        a, b: [N,D] embeddings (paired in order)
        A: [N,N] similarity matrix
        A2: [N,N] similarity matrix for images
        """
        if l2_normalize:
            a = F.normalize(a, dim=-1)
            b = F.normalize(b, dim=-1)
        
        N = a.size(0)
        tau = self.temperature
        
        # Build targets from A
        if A is None or self.hard_clip:
            P = torch.eye(N, device=a.device, dtype=a.dtype)
        else:
            P, _ = self._build_targets_from_A(A)
        
        total_loss = 0.0
        
        # Cross-modal loss (a-b) - use InfoNCE-style with numerator/denominator
        if self.cross_weight > 0:
            L_ab = (a @ b.T) / tau
            L_ba = L_ab.T
            L_aa = (a @ a.T) / tau
            L_bb = (b @ b.T) / tau
            
            if slides is not None:
                mask = ~np.equal.outer(slides, slides).astype(np.bool_)
                mask = torch.tensor(mask).to(A.device)
                L_ab *= mask
                L_ba *= mask
                L_aa *= mask
                L_bb *= mask
            loss_ab = self._info_nce_style(L_ab, L_aa, P)
            loss_ba = self._info_nce_style(L_ba, L_bb, P)
            total_loss = loss_ab + loss_ba
        else:
            L_ab = (a @ b.T) / tau
            L_ba = L_ab.T
        
            total_loss = super()._info_nce_style(L_ab, P) + super()._info_nce_style(L_ba, P)

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        elif self.reduction == 'none':
            return total_loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class PearsonCorrelationLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(PearsonCorrelationLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, y_pred, y_true):

        """
        Compute Pearson correlation loss (1 - correlation coefficient)
        
        Args:
            y_pred: predicted values
            y_true: true values
            
        Returns:
            loss: 1 - pearson correlation coefficient
        """

        y_pred_centered = y_pred - torch.mean(y_pred, dim=-1, keepdim=True)
        y_true_centered = y_true - torch.mean(y_true, dim=-1, keepdim=True)
        
        numerator = torch.sum(y_pred_centered * y_true_centered, dim=-1)
        denominator = torch.sqrt(torch.sum(y_pred_centered ** 2, dim=-1) * torch.sum(y_true_centered ** 2, dim=-1))
        
        correlation = numerator / (denominator + 1e-8)
        loss = 1 - correlation
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
        

class NCELoss(nn.Module):
    """
    A class that uses only NCE (Noise Contrastive Estimation) loss for embedding two modalities.
    This is a simplified version focused solely on contrastive learning between two different data types.
    """
    
    def __init__(self, temperature=0.5, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def calc_similarity_batch(self, a, b):
        """
        Calculate cosine similarity between all pairs in the batch.
        
        Args:
            a: First modality embeddings [batch_size, embedding_dim]
            b: Second modality embeddings [batch_size, embedding_dim]
            
        Returns:
            Similarity matrix [2*batch_size, 2*batch_size]
        """
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )
    
    def forward(self, modality_1_features, modality_2_features):
        """
        Compute NCE loss between two modalities.
        
        Args:
            modality_1_features: Features from first modality [batch_size, embedding_dim]
            modality_2_features: Features from second modality [batch_size, embedding_dim]
            
        Returns:
            NCE loss value (scalar tensor)
        """
        batch_size = modality_1_features.shape[0]
        
        # Create mask to exclude self-similarities
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        
        # Normalize features to unit sphere
        z_i = F.normalize(modality_1_features, p=2, dim=1)
        z_j = F.normalize(modality_2_features, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = self.calc_similarity_batch(z_i, z_j)
        
        # Extract positive pairs (corresponding indices)
        sim_ij = torch.diag(similarity_matrix, batch_size)  # modality_1 -> modality_2
        sim_ji = torch.diag(similarity_matrix, -batch_size)  # modality_2 -> modality_1
        
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # Calculate numerator (positive pairs)
        numerator = torch.exp(positives / self.temperature)
        
        # Calculate denominator (all pairs except self)
        denominator = self.mask.to(similarity_matrix.device) * torch.exp(
            similarity_matrix / self.temperature
        )
        
        # Compute NCE loss
        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")



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

        # Convert labels to {0,1} if needed
        if y_true.min() < 0:
            # assume {-1,1}
            y_true01 = (y_true > 0).float()
        else:
            y_true01 = y_true.float()

        pos_mask = (y_true01 == 1)
        neg_mask = ~pos_mask

        # Safety: avoid empty sets (e.g. highly imbalanced tiny batch)
        if not pos_mask.any() or not neg_mask.any():
            # you may prefer to raise an error instead
            return torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)

        pos_scores = y_pred[pos_mask]
        neg_scores = y_pred[neg_mask]

        # Means (a ≈ mean_pos, b ≈ mean_neg)
        mean_pos = pos_scores.mean()
        mean_neg = neg_scores.mean()

        # Variances (A1 and A2 terms)
        var_pos = ((pos_scores - mean_pos) ** 2).mean()
        var_neg = ((neg_scores - mean_neg) ** 2).mean()

        # Margin (squared hinge) term: (m + mean_neg - mean_pos)_+^2
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
