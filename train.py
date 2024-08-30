import config
import factory
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import DataModel
import numpy as np
import random
import logging
logging.basicConfig(filename='training.log', level=logging.INFO)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

def TrainBall(model, inputs, targets, epoch_num = 1000, batch_size = 64, learning_rate=1e-3, device='cpu'):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    dataset = TensorDataset(*model.process_inputs(inputs, targets))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            optimizer.zero_grad() 
            outputs = model(batch_inputs)
            # outputs = outputs.view(outputs.size(0), model.output_size, model.num_classes)
            # loss = criterion(outputs.view(-1, outputs.size(2)), batch_targets.view(-1))
            outputs = outputs.view(-1, model.num_classes)  # Adjust output shape
            loss = criterion(outputs, batch_targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(dataloader))
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f'[{epoch + 1}/{epoch_num}], Loss: {avg_loss:.4f}, learning rate: {scheduler.get_last_lr()[0]}')
            logging.info(f'[{epoch + 1}/{epoch_num}], Loss: {avg_loss:.4f}, learning rate: {scheduler.get_last_lr()[0]}')
    # checkpoint = {
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }
    # torch.save(checkpoint, 'checkpoint.pt')

def TestBall(model, inputs, targets, device='cpu'):
    model.to(device)
    batch_size = 1
    dataset = TensorDataset(*model.process_inputs(inputs, targets))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    correct = 0
    total = 0
    for batch_inputs, batch_targets in dataloader:
        model.eval()
        with torch.no_grad():
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            outputs = model(batch_inputs)
            _, predicted = torch.max(outputs, 1)
            total += batch_targets.size(0)
            # correct += (predicted == batch_targets.reshape(-1)).sum().item()
            correct += (predicted == batch_targets.view(-1)).sum().item()
        # TrainBall(model, batch_inputs, batch_targets, epoch_num = 10, batch_size=batch_size, learning_rate=1e-4, device=device)
    accuracy = correct / total if total > 0 else 0.0
    print(f'Test Accuracy: {accuracy:.4f}')

def Train(models, epoch_num = 1000, batch_size = 64, learning_rate=1e-3, window_sizes=[6,12,24,36,72,144]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for model_info in models:
        raw_inputs, raw_targets = factory.load_data_from_config(model_info)
        print("Model:", model_info["name"])
        for window_size in window_sizes:    
            with factory.create_model_from_config(model_info) as model:
                print(f"{window_size}")
                file_path = f"models/{model_info["name"]}_{window_size}.pth"  
                try:
                    model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
                except:
                    print(f"Create new {file_path}")
                inputs, targets = DataModel.prepare_data_all(raw_inputs, raw_targets, window_size, model.input_size, model.output_size)
                TrainBall( model=model, inputs=inputs, targets=targets, epoch_num = epoch_num, batch_size=batch_size, learning_rate=learning_rate, device=device)
                torch.save(model.state_dict(), file_path)
                # train_inputs, train_targets, test_inputs, test_targets = DataModel.prepare_data(raw_inputs, raw_targets, window_size, model.input_size, model.output_size, 0.95)
                # TrainBall( model=model, inputs=train_inputs, targets=train_targets, epoch_num = epoch_num, batch_size=batch_size, learning_rate=learning_rate, device=device)
                # TestBall( model=model, inputs=test_inputs, targets=test_targets, device=device)
                # torch.save(model.state_dict(), file_path)


if __name__ == "__main__":
    model_names = factory.model_list(config.models)
    parser = argparse.ArgumentParser(description="Train arguments")
    parser.add_argument("-n", "--epoch_num", type=int, help="Train Epoch Number", default=500)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=64)
    parser.add_argument("-l", "--learning_rate", type=float, help="Learning Rate", default=1e-3)
    parser.add_argument(
        "-w",
        "--window_sizes",
        type=int,
        nargs="+",
        help="Window Sizes",
        default=[12],
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        nargs="+",
        choices=model_names,
        default=model_names,
        help="Model name to train.",
    )
    args = parser.parse_args()
    models = [mode for mode in config.models if mode["name"] in args.models]

    Train(models = models, epoch_num=args.epoch_num, batch_size=args.batch_size, learning_rate=args.learning_rate, window_sizes=args.window_sizes)

