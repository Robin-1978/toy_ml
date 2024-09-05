import torch
import torch.nn as nn


class AccuracyLoss(nn.Module):
    def __init__(self, threshold=0.5, alpha = 0.1):
        super(AccuracyLoss, self).__init__()
        self.threshold = threshold
        self.alpha = alpha
        self.loss = nn.L1Loss()

    def forward(self, predictions, targets):
        # 使用 MSE 作为主要损失函数
        mae_loss = self.loss(predictions, targets)
        # 计算准确率
        hits = torch.abs(predictions - targets) < self.threshold
        hit_rate_loss = 1.0 - hits.float().mean()
        # 损失函数结合准确率（示例：MSE - 准确率）
        combined_loss =  mae_loss + self.alpha * hit_rate_loss
        return combined_loss