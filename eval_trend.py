import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import DataModel
from utils import set_seed, log, PrepareData

from model.lstm_attention import LSTM_Attention

def EvaluateClassifier(model, num_epochs=80, learning_rate=0.01, time_step = 5, split=0.95, device="cpu"):
    df, scaler = DataModel.load_ssq_features(3, 3)
    features=[
        "Ball_1_scale",
        "Ball_2_scale",
        "Ball_3_scale",
        "Ball_4_scale",
        "Ball_5_scale",
        "Ball_6_scale",
        "Ball_7_scale",
        "Ball_7_diff_scale",
        "Ball_7_lag_1_scale",
        "Ball_7_lag_2_scale",
        "Ball_7_lag_3_scale",
        'Ball_7_mean_scale',
        'Ball_7_std_scale',
        'Ball_7_size',
        'Ball_7_odd_even',
    ]
    targets=[
        "Ball_7_diff_scale",
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
    y_train = torch.tensor(train_y).to(device)
    X_test = torch.tensor(test_x, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_y).to(device)
    X_predict = torch.tensor(PX, dtype=torch.float32).to(device)

    model.to(device)
    batch_size = 32

    # criterion = nn.L1Loss()
    # criterion = AccuracyLoss(scaler_diff.transform([[0.5]])[0][0], 0.5)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    dataset = TensorDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
 
            if (epoch + 1) % 1 == 0:
                cpu_y_test = y_test.cpu().numpy()

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
    model = LSTM_Attention(15 , 16, hidden_size=1024, num_layers=8, num_heads=64, dropout=0.2)
    EvaluateClassifier(model, num= 7, num_epochs=500, learning_rate=0.01, time_step = 5, split=0.95, device=device)

if __name__ == "__main__":
    # auto_validate()
    validate()