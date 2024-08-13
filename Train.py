import numpy as np
import torch
import DataModel
import LSTMModel
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import LSTMHyperParameters, LSTMRedBallHyperParameters

def PrepareRedData():
    table = DataModel.LoadData("data/ssq.db")
    data = table[["Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6"]].values
    return data

def PrepareRedWindow(seq_data, window_size):
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

def TrainRedLSTM(epoch_num=1000):
    param = LSTMRedBallHyperParameters()
    data_values = PrepareRedData() - 1  # Adjust for zero-based indexing
    window_sizes = [3, 6, 12, 24, 36, 48, 60]
    
    for window_size in window_sizes:
        print(f"\nWindow size: {window_size}")
        
        # Create inputs and targets
        inputs, targets = PrepareRedWindow(data_values, window_size)
        
        # One-hot encoding
        hot_encodes = torch.nn.functional.one_hot(inputs, num_classes=param.num_classes).float()  # Shape: (num_samples, window_size, num_classes)
        
        model = LSTMModel.LSTMModel(input_size= param.input_size, hidden_size=param.hidden_size, num_classes=param.num_classes, num_layers=param.num_layers, dropout=param.dropout)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        dataset = TensorDataset(hot_encodes, targets)
        batch_size = 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(param.epochs):
            model.train()
            epoch_loss = 0
            for batch_inputs, batch_targets in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_inputs)  # Ensure the model output shape is compatible with targets
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Update the learning rate based on the average loss
            scheduler.step(epoch_loss / len(dataloader))
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {avg_loss:.4f}')
                print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

        torch.save(model.state_dict(), f"LSTMModel_Red_{window_size}.pth")

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

def TrainLSTM(epoch_num=1000):
    param = LSTMHyperParameters()
    data_values = PrepareData() - 1  # Adjust for zero-based indexing
    window_sizes = [3, 6, 12, 24, 36, 48, 60]
    
    for window_size in window_sizes:
        print(f"\nWindow size: {window_size}")
        
        # Create inputs and targets
        inputs, targets = PrepareWindow(data_values, window_size)
        
        # One-hot encoding
        hot_encodes = torch.nn.functional.one_hot(inputs, num_classes=param.num_classes).float()  # Shape: (num_samples, window_size, num_classes)
        
        model = LSTMModel.LSTMModel(input_size= param.input_size, hidden_size=param.hidden_size, num_classes=param.num_classes, num_layers=param.num_layers, dropout=param.dropout)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        dataset = TensorDataset(hot_encodes, targets)
        batch_size = 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(param.epochs):
            model.train()
            epoch_loss = 0
            for batch_inputs, batch_targets in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_inputs)  # Ensure the model output shape is compatible with targets
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Update the learning rate based on the average loss
            scheduler.step(epoch_loss / len(dataloader))
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {avg_loss:.4f}')
                print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

        torch.save(model.state_dict(), f"LSTMModel_{window_size}.pth")

if __name__ == '__main__':
    TrainRedLSTM()
    TrainLSTM()
