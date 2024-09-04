import torch
import torch.nn as nn


class AccuracyLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super(AccuracyLoss, self).__init__()
        self.threshold = threshold
        self.loss = nn.L1Loss()

    def forward(self, predictions, targets):
        # 使用 MSE 作为主要损失函数
        mae_loss = self.loss(predictions, targets)
        # 计算准确率
        errors = torch.abs(predictions - targets)
        loss = (errors >= self.threshold).float().mean()
        # 损失函数结合准确率（示例：MSE - 准确率）
        combined_loss = mae_loss + loss
        return combined_loss