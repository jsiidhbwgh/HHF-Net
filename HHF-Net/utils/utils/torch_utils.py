import torch
import torch.nn as nn

class CombinedSmoothL1WithConfidenceLoss(nn.Module):
    def __init__(self, beta=1.0, weight_smooth=0.6, weight_mse=0.4, weight_confidence=0.2):
        """
        :param beta: Threshold for switching between L1 and L2 in Smooth L1 Loss.
        :param weight_smooth: Weight for Smooth L1 Loss in the combined loss.
        :param weight_mse: Weight for MSE Loss in the combined loss.
        :param weight_confidence: Weight for Confidence Loss in the combined loss.
        """
        super(CombinedSmoothL1WithConfidenceLoss, self).__init__()
        self.beta = beta
        self.weight_smooth = weight_smooth
        self.weight_mse = weight_mse
        self.weight_confidence = weight_confidence
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.confidence_loss = nn.BCELoss(reduction='mean')  # 用于置信度损失

    def smooth_l1_loss(self, input, target):
        diff = torch.abs(input - target)
        loss = torch.where(diff < self.beta,
                           0.5 * diff ** 2 / self.beta,
                           diff - 0.5 * self.beta)
        return loss.mean()

    def forward(self, input, target, confidence_pred, confidence_target):
        # Smooth L1 Loss
        smooth_l1 = self.smooth_l1_loss(input, target)
        
        # MSE Loss
        mse = self.mse_loss(input, target)
        
        # Confidence Loss (用于提升模型的置信度预测能力)
        confidence = self.confidence_loss(confidence_pred, confidence_target)
        
        # Combine the losses with respective weights
        combined_loss = (self.weight_smooth * smooth_l1 
                         + self.weight_mse * mse 
                         + self.weight_confidence * confidence)
        return combined_loss

