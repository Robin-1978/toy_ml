from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def k_fold_cross_validation(model_class, X, y, k=5, epochs=10, lr=0.001):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    fold = 1
    mse_scores = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # 转换为PyTorch数据集和数据加载器
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # 初始化模型
        model = model_class()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 训练模型
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
        
        # 验证模型
        model.eval()
        y_val_pred = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                outputs = model(batch_X)
                y_val_pred.append(outputs.cpu().numpy())
        
        y_val_pred = np.concatenate(y_val_pred, axis=0)
        mse = mean_squared_error(y_val, y_val_pred)
        mse_scores.append(mse)
        
        print(f"Fold {fold}, MSE: {mse:.4f}")
        fold += 1
    
    avg_mse = np.mean(mse_scores)
    print(f"Average MSE over {k} folds: {avg_mse:.4f}")

# 示例模型类
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

# 使用K折交叉验证
k_fold_cross_validation(SimpleModel, X, y)
