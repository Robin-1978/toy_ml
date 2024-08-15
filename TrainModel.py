import DataModel
import SSQModel
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

def TrainBall(model, inputs, targets, epoch_num = 1000, batch_size = 64, learning_rate=1e-3):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    dataset = TensorDataset(torch.tensor(inputs -1), torch.tensor(targets-1))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad() 
            outputs = model(batch_inputs)
            outputs = outputs.view(outputs.size(0), model.input_size, model.num_classes)
            loss = criterion(outputs.view(-1, outputs.size(2)), batch_targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(dataloader))
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {avg_loss:.4f}')
            print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

def TrainSSQ(epoch_num = 1000, batch_size = 64, learning_rate=1e-3):
    reds, blue = DataModel.LoadSSQ()
    for window_size in [6,12,24,36,72,144]:    
        red_train, red_target, blue_train, blue_target = DataModel.PrepareSSQ(window_size, reds, blue)
        print(f"\nSSQ window size: {window_size}")
        print(f"Blue")
        with SSQModel.LSTMEmbedBallModel(input_size= 1, num_classes=16, embedding_size=16, hidden_size=32, num_layers=2, dropout=0.0) as model_blue:
            file_path = f"data/lstm_blue_{window_size}.pth"  
            if os.path.exists(file_path):
                model_blue = torch.load(file_path, weights_only=False)
            TrainBall( model=model_blue, inputs=blue_train, targets=blue_target, epoch_num = epoch_num, batch_size=batch_size, learning_rate=learning_rate)
            torch.save(model_blue, file_path)
        
        print(f"Red") 
        with SSQModel.LSTMEmbedBallModel(input_size= 6, num_classes=33, embedding_size=64, hidden_size=128, num_layers=2, dropout=0.0) as model_red:
            file_path = f"data/lstm_red_{window_size}.pth"
            if os.path.exists(file_path):
                model_red = torch.load(file_path, weights_only=False)
            TrainBall(model=model_red, inputs=red_train, targets=red_target, epoch_num=epoch_num, batch_size=batch_size, learning_rate=learning_rate)
            torch.save(model_red, file_path)

if __name__ == "__main__":
    TrainSSQ(10)