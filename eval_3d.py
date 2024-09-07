import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import DataModel
from model.lstm_cnn import CNN_LSTM_Model
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

def PrepareData(df, features=[], targets=[], window_size=5):
    X = []
    y = []
    for i in range(len(df) - window_size):
        X.append(df[features].iloc[i:i + window_size].values)
        y.append(df[targets].iloc[i + window_size])
    return np.array(X), np.array(y), np.expand_dims(np.array(df[features].iloc[-window_size:]), axis=0)

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

def EvaluateModel(model, num_epochs=80, learning_rate=0.01, time_step = 5, split=0.95, device="cpu"):
    df = DataModel.load_3d_features()
    features=[
        "Ball_1_scale",
        "Ball_2_scale",
        "Ball_3_scale",
        "Ball_1_diff_scale",
        "Ball_2_diff_scale",
        "Ball_3_diff_scale",
        'Ball_1_mean_scale',
        'Ball_2_mean_scale',
        'Ball_3_mean_scale',
        'Ball_1_std_scale',
        'Ball_2_std_scale',
        'Ball_3_std_scale',
    ]
    targets=[
        "Ball_1_trend",
    ]
    X, y, PX = PrepareData(df, features=features, targets=targets, window_size=time_step)
    # y = y - 1
    split_index =int(len(X) * split)
    train_x = X[:split_index]
    train_y = y[:split_index]
    test_x = X[split_index:]
    test_y = y[split_index:]

    # Convert to PyTorch tensors
    X_train = torch.tensor(train_x, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_y.flatten(), dtype=torch.long).to(device)
    X_test = torch.tensor(test_x, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_y.flatten(), dtype=torch.long).to(device)
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

                log(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {eval_loss:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f} Accuracy: {accuracy:.4f}" )
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


def Eval():
    from sklearn.ensemble import VotingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier

    model1 = RandomForestClassifier()
    model2 = GradientBoostingClassifier()
    model3 = MLPClassifier(hidden_layer_sizes=(128,), max_iter=100, random_state=42)
    df = DataModel.load_3d_features()
    features=[
        "Ball_1_scale",
        "Ball_2_scale",
        "Ball_3_scale",
        "Ball_1_diff_scale",
        "Ball_2_diff_scale",
        "Ball_3_diff_scale",
        'Ball_1_mean_scale',
        'Ball_2_mean_scale',
        'Ball_3_mean_scale',
        'Ball_1_std_scale',
        'Ball_2_std_scale',
        'Ball_3_std_scale',
    ]
    targets=[
        "Ball_3",
    ]
    X, y, PX = PrepareData(df, features=features, targets=targets, window_size=1)
    X =  np.squeeze(X)
    y =  np.squeeze(y)
    PX =  PX.reshape(1,-1)
    ensemble_model = VotingClassifier(estimators=[
        ('rf', model1),
        ('gb', model2),
        ('mlp', model3)
    ], voting='soft')

    ensemble_model.fit(X, y)
    y_pred = ensemble_model.predict(PX)
    log(f"Predict: {y_pred}")

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

    # model = LSTM_Attention(12, 10, hidden_size=128, num_layers=3, num_heads=8, dropout=0.2)
    from model.lstm_cnn import CNN_LSTM_Model
    model = CNN_LSTM_Model(
        input_size=12,
        output_size=3,
        hidden_size=128,
        num_layers=3,
        dropout=0.2,
        kernel_size=3,
        cnn_out_channels=[64],
    )
    # from model.ml import MLModel
    # model = MLModel(9, 16, hidden_sizes=[64,128], dropout=0.0)
    EvaluateModel(model, num_epochs=500, learning_rate=0.01, time_step = 12, split=0.95, device=device)

if __name__ == "__main__":
    # auto_validate()
    # Eval()
    validate()