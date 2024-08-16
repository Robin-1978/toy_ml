import DataModel
import Model
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse

def TrainBall(model, inputs, targets, epoch_num = 1000, batch_size = 64, learning_rate=1e-3, device='cpu'):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    input_encode, target_encode = model.process_inputs(inputs, targets)
    dataset = TensorDataset(input_encode, target_encode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
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
            print(f'[{epoch + 1}/{epoch_num}], Loss: {avg_loss:.4f}, learning rate: {scheduler.get_last_lr()[0]}')

def TrainSSQ(epoch_num = 1000, batch_size = 64, learning_rate=1e-3, window_sizes=[6,12,24,36,72,144]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reds, blue = DataModel.LoadSSQ()
    for window_size in window_sizes:    
        red_train, red_target, blue_train, blue_target = DataModel.PrepareSSQ(window_size, reds-1, blue-1)
        print(f"\n{window_size}")

        with Model.LSTMEmbedBallModel(input_size= 1, num_classes=16, embedding_size=16, hidden_size=32, num_layers=2, dropout=0.0) as model_embed_blue:
            file_path = f"data/lstm_embed_blue_{window_size}.pth"  
            try:
                model_embed_blue.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
            except:
                os.remove(file_path)
                print(f"removing {file_path}")

            TrainBall( model=model_embed_blue, inputs=blue_train, targets=blue_target, epoch_num = epoch_num, batch_size=batch_size, learning_rate=learning_rate, device=device)
            torch.save(model_embed_blue.state_dict(), file_path)
        
        with Model.LSTMOneHotBallModel(input_size= 1, num_classes=16, hidden_size=32, num_layers=2, dropout=0.0) as model_blue:
            file_path = f"data/lstm_onehot_blue_{window_size}.pth"  
            try:
                model_blue.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
            except:
                os.remove(file_path)
                print(f"removing {file_path}")
            
            TrainBall( model=model_blue, inputs=blue_train, targets=blue_target, epoch_num = epoch_num, batch_size=batch_size, learning_rate=learning_rate, device=device)
            torch.save(model_blue.state_dict(), file_path)

        with Model.LSTMEmbedBallModel(input_size= 6, num_classes=33, embedding_size=64, hidden_size=128, num_layers=2, dropout=0.0) as model_red:
            file_path = f"data/lstm_embed_red_{window_size}.pth"
            try:
                model_red.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
            except:
                os.remove(file_path)
                print(f"removing {file_path}")

            TrainBall(model=model_red, inputs=red_train, targets=red_target, epoch_num=epoch_num, batch_size=batch_size, learning_rate=learning_rate, device=device)
            torch.save(model_red.state_dict(), file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('-n', '--epoch_num', type=int, help = 'Train Epoch Number', default=10)
    parser.add_argument('-b', '--batch_size', type=int, help = 'Batch Size', default=64)
    parser.add_argument('-l', '--learning_rate', type=float, help = 'Learning Rate', default=1e-3)
    parser.add_argument('-w', '--window_sizes', type=int, nargs='+', help = 'Window Sizes', default=[3,6,12,24,36,72,144])
    parser.add_argument(
        '-m', '--model',
        type=str,
        nargs='+',
        choices=['LSTMOneHotRed','LSTMOneHotBlue' 'LSTMEmbedRed', 'LSTMEmbedBlue'],
        default=['LSTMOneHotRed','LSTMOneHotBlue' 'LSTMEmbedRed', 'LSTMEmbedBlue'],
        help='Model name to train.'
    )
    args = parser.parse_args()
    TrainSSQ(epoch_num=args.epoch_num, batch_size=args.batch_size, learning_rate=args.learning_rate, window_sizes=args.window_sizes)
