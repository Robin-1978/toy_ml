from sympy import factor
import config
import factory
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import DataModel

def TrainBall(model, inputs, targets, epoch_num = 1000, batch_size = 64, learning_rate=1e-3, device='cpu'):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
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
            outputs = outputs.view(outputs.size(0), model.input_size, model.num_classes)
            loss = criterion(outputs.view(-1, outputs.size(2)), batch_targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(dataloader))
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f'[{epoch + 1}/{epoch_num}], Loss: {avg_loss:.4f}, learning rate: {scheduler.get_last_lr()[0]}')


def Train(models, epoch_num = 1000, batch_size = 64, learning_rate=1e-3, window_sizes=[6,12,24,36,72,144]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for model_info in models:
        with factory.create_model_from_config(model_info) as model:
            data = factory.load_data_from_config(model_info)
            print("Model:", model_info["name"])
            for window_size in window_sizes:    
                print(f"{window_size}")
                file_path = f"data/{model_info["name"]}_{window_size}.pth"  
                try:
                    model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
                except:
                    print(f"Create new {file_path}")
                inputs, targets = DataModel.prepare_data(data, window_size, model.input_size)
                TrainBall( model=model, inputs=inputs, targets=targets, epoch_num = epoch_num, batch_size=batch_size, learning_rate=learning_rate, device=device)
                torch.save(model.state_dict(), file_path)


if __name__ == "__main__":
    model_names = factory.model_list(config.models)
    parser = argparse.ArgumentParser(description="Train arguments")
    parser.add_argument("-n", "--epoch_num", type=int, help="Train Epoch Number", default=10)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=64)
    parser.add_argument("-l", "--learning_rate", type=float, help="Learning Rate", default=1e-3)
    parser.add_argument(
        "-w",
        "--window_sizes",
        type=int,
        nargs="+",
        help="Window Sizes",
        default=[3, 6, 12, 24, 36, 72, 144],
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

