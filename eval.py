from re import A
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

import DataModel
from model.lstm_cnn import CNN_LSTM_Model
from model.lstm_cnn import HyperParameters as HP_CNN

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

def TrendScore(predic_y, target_y):
    pass


def EvaluateModel(model, hp, num, num_epochs=100, learning_rate=0.01, time_step = 5,device="cpu"):
    balls, diff = DataModel.load_ssq_single_diff(num)
    diff_data = diff.dropna().values
    balls_data = balls[1:].to_numpy() - 1

    scaler_ball = MinMaxScaler(feature_range=(0, 1))
    scaled_ball_data = scaler_ball.fit_transform(balls_data.reshape(-1, 1))

    scaler_diff = MinMaxScaler(feature_range=(0, 1))
    scaled_diff_data = scaler_diff.fit_transform(diff_data.reshape(-1, 1))

    # time_step = 5  # Number of time steps to look back

    scaled_data = np.column_stack((scaled_ball_data, scaled_diff_data))

    # X, y = create_dataset_single(scaled_data, time_step)
    X, y = create_dataset(scaled_data, scaled_diff_data, time_step)

    tscv = TimeSeriesSplit(n_splits=5)
    best_global_hits = 0
    best_hits_state = None
    best_global_trend = 0
    best_trend_state = None
    all_final_hits = []
    all_final_trend = []
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        # log(f'TRAIN: {train_index} TEST:, {test_index}')
        log(f"Fold {fold+1}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test).to(device)

        model.to(device)
        # num_epochs = 200
        batch_size = 32

        criterion = nn.L1Loss()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
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
                outputs, hidden = model(batch_X)
                hidden = (hidden[0].detach(), hidden[1].detach())
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Step scheduler after each epoch
            loss = epoch_loss / len(data_loader)
            scheduler.step(loss)

            if (epoch + 1) % 100 == 0:  # Print every 5 epochs
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
                scaled_outputs = scaler_diff.inverse_transform(outputs.detach().cpu().numpy())
                scaled_outputs = torch.tensor(scaled_outputs)
                scaled_ytest = scaler_diff.inverse_transform(y_test.detach().cpu().numpy())
                scaled_ytest = torch.tensor(scaled_ytest)
                same_trend = (
                    (
                        ((scaled_ytest * scaled_outputs) > 0)
                        | ((scaled_ytest == 0) & (scaled_outputs == 0))
                    )
                    .sum()
                    .item()
                )
                hits = (torch.abs(scaled_ytest - scaled_outputs) < 0.5).sum().item()

                if (epoch + 1) % 100 == 0:  # Print every 5 epochs
                    log(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {eval_loss:.4f}")
                if same_trend > best_trend:
                    best_trend = same_trend
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Test Same Trend: {same_trend}/{len(y_test)} ({same_trend/len(y_test) * 100:.2f}%)"
                    )
                if best_trend > best_global_trend:
                    best_global_trend = best_trend
                    best_trend_state = model.state_dict()
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Test Best Global Trend: {best_trend}/{len(y_test)} ({best_trend/len(y_test) * 100:.2f}%)"
                    )

                if hits > best_hits:
                    best_hits = hits
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Test Hits: {hits}/{len(y_test)} ({hits/len(y_test) * 100:.2f}%)"
                    )
                if best_global_hits < best_hits:
                    best_global_hits = best_hits
                    best_hits_state = model.state_dict()
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Test Best Global Hits: {best_hits}/{len(y_test)} ({best_hits/len(y_test) * 100:.2f}%)"
                    )

        log(
            f"{fold+1} Lastest train loss: {loss:.4f} test loss: {eval_loss:.4f} test hits: {hits}/{len(y_test)} ({hits/len(y_test) * 100:.2f}%) test same trend: {same_trend}/{len(y_test)} ({same_trend/len(y_test) * 100:.2f}%)"
        )
        all_final_hits.append(hits/len(y_test))
        all_final_trend.append(same_trend/len(y_test))

    SaveModel(best_trend_state, hp, f"models/best_trend_state_{num}_{best_global_trend}")
    SaveModel(best_hits_state, hp, f"models/best_hits_state_{num}_{best_global_hits}")
    print(
        f"Best Hits: {best_global_hits}/{len(y_test)} ({best_global_hits/len(y_test) * 100:.2f}%) Best Trend: {best_global_trend}/{len(y_test)} ({best_global_trend/len(y_test) * 100:.2f}%)"
    )
    return  np.mean(all_final_hits),  np.mean(all_final_trend)


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


    hpcnn = HP_CNN(2, 1, 96, 2, 0.2, 3, 32)
    model = CNN_LSTM_Model(
        input_size=hpcnn.input_size,
        output_size=hpcnn.output_size,
        hidden_size=hpcnn.hidden_size,
        num_layers=hpcnn.num_layers,
        dropout=hpcnn.dropout,
        kernel_size=hpcnn.kernel_size,
        cnn_out_channels=hpcnn.cnn_out_channels,
    )
    EvaluateModel(model, hpcnn, 7, 100, "cpu")


import optuna

def objective_best_hits(trial):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = trial.suggest_categorical('hidden_size', [8, 16, 32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 9)
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
    hits, trend = EvaluateModel(model, hpcnn, 7, num_epochs, learning_rate, time_step, device)
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
    hits, trend = EvaluateModel(model, hpcnn, 7, num_epochs, learning_rate, time_step, device)
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
    auto_validate()