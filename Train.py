import numpy as np
import torch
import DataModel
import SSQModel
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import LSTMHyperParameters, LSTMRedBallHyperParameters

def PrepareRedData():
    table = DataModel.LoadData("data/ssq.db")
    data = table[["Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6"]].values
    return data

def PrepareRedWindow(seq_data, window_size):
    if seq_data is None or window_size < 1:
        raise ValueError("seq_data and window_size must not be None or less than 1")

    num_samples = len(seq_data) - window_size
    
    # Initialize numpy arrays with the correct size
    inputs_array = np.zeros((num_samples, window_size, 6), dtype=np.int64)
    targets_array = np.zeros((num_samples, 6), dtype=np.int64)

    for i in range(num_samples):
        start = i
        end = i + window_size
        inputs_array[i] = seq_data[start:end]
        targets_array[i] = seq_data[end]
    
    # Convert numpy arrays to PyTorch tensors
    inputs_tensor = torch.tensor(inputs_array, dtype=torch.long)
    targets_tensor = torch.tensor(targets_array, dtype=torch.long)
    
    return inputs_tensor, targets_tensor
# def PrepareRedWindow(seq_data, window_size):
#     inputs, targets = [], []
#     for i in range(len(seq_data) - window_size):
#         # Convert numpy arrays to PyTorch tensors
#         inputs.append(torch.tensor(seq_data[i:i + window_size], dtype=torch.long))
#         targets.append(seq_data[i + window_size])
    
#     # Stack tensors into a single tensor for inputs
#     inputs_tensor = torch.stack(np.array(inputs))  # Shape: (num_samples, window_size)
    
#     # Convert targets to a tensor
#     targets_tensor = torch.tensor(np.array(targets), dtype=torch.long)  # Shape: (num_samples,)
    
#     return inputs_tensor, targets_tensor

def TrainRedLSTM(epoch_num):
    param = LSTMRedBallHyperParameters()
    data_values = PrepareRedData() - 1  # Adjust for zero-based indexing
    window_sizes = param.window_sizes
    
    for window_size in window_sizes:
        print(f"\nWindow size: {window_size}")
        
        # Create inputs and targets
        inputs, targets = PrepareRedWindow(data_values, window_size)
        
        # One-hot encoding
        # hot_encodes = torch.nn.functional.one_hot(inputs, num_classes=param.num_classes).float()  # Shape: (num_samples, window_size, num_classes)
        
        model = SSQModel.LSTMRedModel(input_size= param.input_size, embedding_size=param.embedding_size, hidden_size=param.hidden_size, num_classes=param.num_classes, num_layers=param.num_layers, dropout=param.dropout)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        dataset = TensorDataset(inputs, targets)
        batch_size = param.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(param.epochs):
            model.train()
            epoch_loss = 0
            for batch_inputs, batch_targets in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_inputs)  # Ensure the model output shape is compatible with targets
                outputs = outputs.view(outputs.size(0), param.input_size, param.num_classes)
                loss = criterion(outputs.view(-1, outputs.size(2)), batch_targets.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Update the learning rate based on the average loss
            scheduler.step(epoch_loss / len(dataloader))
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {avg_loss:.4f}')
                print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

        torch.save(model.state_dict(), f"data/{epoch_num}/lstm_red_{window_size}.pth")

def PrepareData():
    table = DataModel.LoadData("data/ssq.db")
    data = table["Ball_7"].values
    return data

def PrepareWindow(seq_data, window_size):
    inputs, targets = [], []
    for i in range(len(seq_data) - window_size):
        # Convert numpy arrays to PyTorch tensors
        inputs.append(torch.tensor(seq_data[i:i + window_size], dtype=torch.long))
        targets.append(seq_data[i + window_size])
    
    # Stack tensors into a single tensor for inputs
    inputs_tensor = torch.stack(inputs)  # Shape: (num_samples, window_size)
    
    # Convert targets to a tensor
    targets_tensor = torch.tensor(targets, dtype=torch.long)  # Shape: (num_samples,)
    
    return inputs_tensor, targets_tensor

def TrainLSTM(epoch_num):
    param = LSTMHyperParameters()
    data_values = PrepareData() - 1  # Adjust for zero-based indexing
    window_sizes = param.window_sizes
    
    for window_size in window_sizes:
        print(f"\nWindow size: {window_size}")
        
        # Create inputs and targets
        inputs, targets = PrepareWindow(data_values, window_size)
        
        # One-hot encoding
        hot_encodes = torch.nn.functional.one_hot(inputs, num_classes=param.num_classes).float()  # Shape: (num_samples, window_size, num_classes)
        
        model = SSQModel.LSTMBlueModel(input_size= param.input_size, hidden_size=param.hidden_size, num_classes=param.num_classes, num_layers=param.num_layers, dropout=param.dropout)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Using device: {device}")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        dataset = TensorDataset(hot_encodes, targets)
        batch_size = param.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(param.epochs):
            model.train()
            epoch_loss = 0
            for batch_inputs, batch_targets in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_inputs)  # Ensure the model output shape is compatible with targets
                loss = criterion(outputs.view, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Update the learning rate based on the average loss
            scheduler.step(epoch_loss / len(dataloader))
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {avg_loss:.4f}')
                print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

        torch.save(model.state_dict(), f"data/{epoch_num}/lstm_blue_{window_size}.pth")

if __name__ == '__main__':
    param = LSTMHyperParameters()
    TrainRedLSTM(param.epochs)
    # TrainLSTM(param.epochs)
