import DataModel
import model
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import factory
import argparse
import config

def PredictBall(model, inputs, device='cpu'):
    model.eval()
    inputs, _ = model.process_inputs(inputs, [])
    inputs = inputs.reshape(inputs.size(0), -1).unsqueeze(0)
    with torch.no_grad():
        output = model(inputs)
        output = output.view(output.size(0), model.input_size, model.num_classes)
        output = output.view(-1, output.size(2))
        softmax = torch.softmax(output, dim=1)
        # predicted_class = torch.argmax(output, dim=1)
        predicted_numbers = (torch.topk(softmax, k=3, dim=1).indices + 1)
        # predicted_numbers = (torch.topk(torch.softmax(output, dim=1), 3).indices + 1).reshape(-1)
    print(f"Predicted: {predicted_numbers.tolist()}\n")
    return predicted_numbers

def Predict(models, window_sizes):
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
                    print(f"Failed to load trained data {file_path}")
                    break

                PredictBall( model=model, inputs=data[-window_size:], device=device)

if __name__ == "__main__":
    models = factory.model_list(config.models)
    parser = argparse.ArgumentParser(description="Predict arguments")
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
        choices=models,
        default=models,
        help="Model name to train.",
    )
    args = parser.parse_args()
    models = [mode for mode in config.models if mode["name"] in args.models]

    Predict(models = models,  window_sizes=args.window_sizes)
