import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        
        Args:
        - alpha (list or tensor): Class weights to address imbalance. If None, weights are not applied.
        - gamma (float): Focusing parameter to adjust the loss contribution of easy examples.
        - reduction (str): Specifies the reduction to apply: 'none', 'mean', or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
 
        probs = F.softmax(logits, dim=1)
        
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        p_t = (probs * targets_one_hot).sum(dim=1)
        
        focal_factor = (1 - p_t) ** self.gamma
        
        log_p_t = torch.log(p_t + 1e-8) 
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = -alpha_t * focal_factor * log_p_t
        else:
            loss = -focal_factor * log_p_t
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 