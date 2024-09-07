from re import A
from sympy import N
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import random
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import DataModel
from model.lstm_cnn import CNN_LSTM_Model
from model.lstm_cnn import HyperParameters as HP_CNN
from model.accuracy_loss import AccuracyLoss
from model.lstm_attention import LSTM_Attention

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log(message):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}")

def calculate_lag_features(window, lag):
    # 确保滞后期数不超过窗口大小
    lag = min(lag, len(window) - 1)
    # 获取滞后特征
    lag_features = [window[i] for i in range(lag, 0, -1)]
    return np.array(lag_features).flatten()  # 确保是二维数组转换为一维数组

def calculate_features(window):
    # 计算统计特征
    return {
        'std_dev': np.std(window),
        'max': np.max(window),
        'min': np.min(window),
        'mean': np.mean(window)
    }

def create_feature_matrix(data, window_size, lag):
    num_samples = len(data) - window_size + 1
    feature_matrix = []

    for i in range(num_samples):
        window = data[i:i + window_size]
        
        # 计算滞后特征
        lag_features = calculate_lag_features(window, lag)
        
        # 计算统计特征
        stats = calculate_features(window)
        stats_features = [stats['std_dev'], stats['max'], stats['min'], stats['mean']]
        
        # 将统计特征转换为一维数组
        stats_features = np.array(stats_features)
        
        # 检查特征的形状
        # print(f"lag_features shape: {lag_features.shape}")
        # print(f"stats_features shape: {stats_features.shape}")
        
        # 合并特征
        features = np.hstack((lag_features, stats_features))
        
        feature_matrix.append(features)
    
    return np.array(feature_matrix)

def add_original_features(original_features, stats_features):
    # 合并原始特征和统计特征
    return np.hstack((original_features, stats_features))

def create_dataset(X, y, time_step=1):
    train_X, target_y = [], []
    for i in range(len(X) - time_step):
        train_X.append(X[i : (i + time_step)])
        target_y.append(y[i + time_step])
    return np.array(train_X), np.array(target_y)

def SaveModel(state_dict, hp, name):
    import json

    torch.save(state_dict, name + ".pth")
    with open(name + ".json", "w") as f:
        json.dump(hp.to_dict(), f, indent=4)

def PrepareData(df, features=[], targets=[], window_size=5):
    X = []
    y = []
    for i in range(len(df) - window_size):
        X.append(df[features].iloc[i:i + window_size].values)
        y.append(df[targets].iloc[i + window_size])
    return np.array(X), np.array(y), np.expand_dims(np.array(df[features].iloc[-window_size:]), axis=0)

def calculate_trend_accuracy(y_true, y_pred):
    # Convert to numpy arrays for easier manipulation
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    # Calculate the trend for actual values
    true_trend = np.diff(y_true)
    # Calculate the trend for predicted values
    pred_trend = np.diff(y_pred)

    # Determine the trend direction
    true_trend_direction = np.sign(true_trend)
    pred_trend_direction = np.sign(pred_trend)

    # Calculate the trend accuracy
    trend_accuracy = np.mean(true_trend_direction == pred_trend_direction)

    return trend_accuracy

def EvaluateModel(model, num, num_epochs=80, learning_rate=0.01, time_step = 5, split=0.95, device="cpu"):
    df = DataModel.load_ssq_features(3, 3)
    features=[
        "Ball_7_scale",
        "Ball_7_diff_scale",
        "Ball_7_lag_1_scale",
        "Ball_7_lag_2_scale",
        "Ball_7_lag_3_scale",
        # "Ball_7_freq_scale",
        'Ball_7_mean_scale',
        'Ball_7_std_scale',
        'Ball_7_size',
        'Ball_7_odd_even',
    ]
    targets=[
        "Ball_7",
    ]
    X, y, PX = PrepareData(df, features=features, targets=targets, window_size=time_step)
    y = y - 1
    split_index =int(len(X) * split)
    train_x = X[:split_index]
    train_y = y[:split_index]
    test_x = X[split_index:]
    test_y = y[split_index:]

    # Convert to PyTorch tensors
    X_train = torch.tensor(train_x, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_y.flatten()).to(device)
    X_test = torch.tensor(test_x, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_y.flatten()).to(device)
    X_predict = torch.tensor(PX, dtype=torch.float32).to(device)

    model.to(device)
    batch_size = 32

    # criterion = nn.L1Loss()
    # criterion = AccuracyLoss(scaler_diff.transform([[0.5]])[0][0], 0.5)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    dataset = TensorDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    best_hits = 0
    
    best_trend = 0
    for epoch in range(num_epochs):
        model.train()
        hidden = None
        epoch_loss = 0
        for batch_X, batch_y in data_loader:
            optimizer.zero_grad()   
            if hidden is not None:
                h, c = hidden
                if h.size(1) != batch_X.size(0):  
                    h = h[:, :batch_X.size(0), :].contiguous()
                    c = c[:, :batch_X.size(0), :].contiguous()
                    hidden = (h, c)
            outputs, hidden = model(batch_X, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Step scheduler after each epoch
        loss = epoch_loss / len(data_loader)
        scheduler.step(loss)

        if (epoch + 1) % 1 == 0:  # Print every 5 epochs
            log(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss:.4f} learning rate: {scheduler.get_last_lr()[0]}"
            )

        if scheduler.get_last_lr()[0] < 1e-5:
            log(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss:.4f} learning rate: {scheduler.get_last_lr()[0]}"
            )
            break

        with torch.no_grad():
            model.eval()
            outputs, _= model(X_test)
            eval_loss = criterion(outputs, y_test)
            predicted_classes = torch.argmax(outputs, dim=1)
            # scaled_ytest = torch.tensor(scaled_ytest)

            hits = (predicted_classes == y_test).sum().item()
 
            if (epoch + 1) % 1 == 0:
                # 计算精确率 (Precision)
                precision = precision_score(y_test, predicted_classes, average='macro', zero_division=0)  # 或 'micro', 'weighted'
                # 计算召回率 (Recall)
                recall = recall_score(y_test, predicted_classes, average='macro', zero_division=0)  # 或 'micro', 'weighted'
                # 计算 F1 分数
                f1 = f1_score(y_test, predicted_classes, average='macro', zero_division=0)  # 或 'micro', 'weighted'
                # 计算准确率 (Accuracy)（通常用于多分类任务）
                accuracy = accuracy_score(y_test, predicted_classes)

                trend = calculate_trend_accuracy(y_test, predicted_classes)

                log(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {eval_loss:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f} Accuracy: {accuracy:.4f} Trend: {trend:.4f}" )
            # if same_trend > best_trend:
            #     best_trend = same_trend
            #     print(
            #         f"Epoch [{epoch+1}/{num_epochs}], Test Same Trend: {same_trend}/{len(y_test)} ({same_trend/len(y_test) * 100:.2f}%)"
            #     )

            if hits > best_hits:
                best_hits = hits
                predicts, _ = model(X_predict)
                predicts = torch.argmax(predicts, dim=1)
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Best Test Hits: {hits}/{len(y_test)} ({hits/len(y_test) * 100:.2f}%) Predict:{predicts+1}"
                )
    return  hits

def validate():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.multiprocessing.set_start_method("spawn")
    torch.autograd.set_detect_anomaly(True)

    print(f"Using device: {device}")
    import argparse

    parser = argparse.ArgumentParser(description="SSQ arguments")
    # parser.add_argument("-n", "--epoch_num", type=int, help="Train Epoch Number", default=500)
    parser.add_argument("-b", "--ball_num", type=int, help="Ball Number", default=7)
    parser.add_argument("-p", "--predict_num", type=int, help="Predict Number", default=0)
    args = parser.parse_args()

    
    # hpcnn = HP_CNN(9, 16, hidden_size=512, num_layers=5, dropout=0.2, kernel_size=3, cnn_out_channels=[64])
    # model = CNN_LSTM_Model(
    #     input_size=hpcnn.input_size,
    #     output_size=hpcnn.output_size,
    #     hidden_size=hpcnn.hidden_size,
    #     num_layers=hpcnn.num_layers,
    #     dropout=hpcnn.dropout,
    #     kernel_size=hpcnn.kernel_size,
    #     cnn_out_channels=hpcnn.cnn_out_channels,
    # )
    model = LSTM_Attention(9, 16, hidden_size=128, num_layers=2, num_heads=8, dropout=0.1)

    # from model.ml import MLModel
    # model = MLModel(9, 16, hidden_sizes=[64,128], dropout=0.0)
    EvaluateModel(model, num= 7, num_epochs=500, learning_rate=0.01, time_step = 5, split=0.95, device=device)


import optuna

def objective_best_hits(trial):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = trial.suggest_categorical('hidden_size', [8, 16, 32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 5)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    num_epochs = trial.suggest_int('num_epochs', 1, 300)
    # kernal_size = trial.suggest_int('kernal_size', 1, 9)
    out_channels = trial.suggest_int('out_channels', 2, 128)
    time_step = trial.suggest_int('time_step', 3, 128)
    hpcnn = HP_CNN(2, 1, hidden_size, num_layers, dropout, 3, out_channels)
    model = CNN_LSTM_Model(
        input_size=hpcnn.input_size,
        output_size=hpcnn.output_size,
        hidden_size=hpcnn.hidden_size,
        num_layers=hpcnn.num_layers,
        dropout=hpcnn.dropout,
        kernel_size=hpcnn.kernel_size,
        cnn_out_channels=hpcnn.cnn_out_channels,
    )
    hits, trend = EvaluateModel(model, hpcnn, 7, num_epochs, learning_rate, time_step, 0.95,device)
    return hits

def objective_best_trend(trial):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = trial.suggest_categorical('hidden_size', [8, 16, 32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 9)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    num_epochs = trial.suggest_int('num_epochs', 10, 300)
    # kernal_size = trial.suggest_int('kernal_size', 1, 7)
    out_channels = trial.suggest_int('out_channels', 16, 128)
    time_step = trial.suggest_int('time_step', 3, 128)
    hpcnn = HP_CNN(2, 1, hidden_size, num_layers, dropout, 3, out_channels)
    model = CNN_LSTM_Model(
        input_size=hpcnn.input_size,
        output_size=hpcnn.output_size,
        hidden_size=hpcnn.hidden_size,
        num_layers=hpcnn.num_layers,
        dropout=hpcnn.dropout,
        kernel_size=hpcnn.kernel_size,
        cnn_out_channels=hpcnn.cnn_out_channels,
    )
    hits, trend = EvaluateModel(model, hpcnn, 7, num_epochs, learning_rate, time_step, 0.95, device)
    return trend

def auto_validate():
    study_best_hits = optuna.create_study(direction='maximize')
    study_best_hits.optimize(objective_best_hits, n_trials=100)

    print("Best parameters for Best Hits:", study_best_hits.best_params)
    print("Best Best Hits value:", study_best_hits.best_value)

    study_best_trend = optuna.create_study(direction='maximize')
    study_best_trend.optimize(objective_best_trend, n_trials=100)

    print("Best parameters for Best Trend:", study_best_trend.best_params)
    print("Best Best Trend value:", study_best_trend.best_value)


if __name__ == "__main__":
    # auto_validate()
    validate()