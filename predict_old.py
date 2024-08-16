import torch
import DataModel
import model
from config_old import LSTMHyperParameters, LSTMRedBallHyperParameters

def PrepareData():
    table = DataModel.LoadData("data/ssq.db")
    data = table["Ball_7"].values
    return data

def PrepareRedData():
    table = DataModel.LoadData("data/ssq.db")
    data = table[["Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6"]].values
    return data

def PrepareWindow(seq_data, window_size):
    inputs = torch.tensor(seq_data[-window_size:], dtype=torch.long).unsqueeze(0)  # Shape: (1, window_size, num_classes)
    # print (inputs)
    return inputs

def PrepareRedWindow(seq_data, window_size):    
    # Convert numpy arrays to PyTorch tensors
    inputs_tensor = torch.tensor(seq_data[-window_size:], dtype=torch.long).unsqueeze(0)
    
    return inputs_tensor

def Predict(epoch_num, window_size):
    param = LSTMHyperParameters()
    # torch.save(model.state_dict(), f"LSTMModel_{window_size}.pth")
    model = model.LSTMBlueModel(input_size=param.input_size, hidden_size=param.hidden_size, num_classes=param.num_classes, num_layers=param.num_layers, dropout=param.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    model.load_state_dict(torch.load(f"data/{epoch_num}/lstm_blue_{window_size}.pth", weights_only=True))
    model.eval()
    inputs = PrepareWindow(PrepareData(), window_size) - 1
    hot_encodes = torch.nn.functional.one_hot(inputs, num_classes=16).float()
    print(f"Window Size: {window_size}")
    print(f"Inputs: {(inputs+1).reshape(-1).tolist()}")

    with torch.no_grad():
        output = model(hot_encodes)
        # predicted_class = torch.argmax(output, dim=1)
        predicted_numbers = (torch.topk(torch.softmax(output, dim=1), 3).indices + 1).reshape(-1)
        # predicted_number = predicted_class.item() + 1  # 转换回1到16

    print(f"Predicted: {predicted_numbers.tolist()}\n")
    return predicted_numbers

def PredictRed(epoch_num, window_size):
    param = LSTMRedBallHyperParameters()
    # torch.save(model.state_dict(), f"LSTMModel_{window_size}.pth")
    model = model.LSTMRedModel(input_size=param.input_size, embedding_size=param.embedding_size, hidden_size=param.hidden_size, num_classes=param.num_classes, num_layers=param.num_layers, dropout=param.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    model.load_state_dict(torch.load(f"data/{epoch_num}/lstm_red_{window_size}.pth", weights_only=True))
    model.eval()
    inputs = PrepareRedWindow(PrepareRedData(), window_size) - 1
    # hot_encodes = torch.nn.functional.one_hot(inputs, num_classes=16).float()
    print(f"Window Size: {window_size}")
    print(f"Inputs: {(inputs+1).reshape(-1, inputs.size(2)).tolist()}")

    with torch.no_grad():
        output = model(inputs)
        output = output.view(output.size(0), param.input_size, param.num_classes)
        output = output.view(-1, output.size(2))
        softmax = torch.softmax(output, dim=1)
        # predicted_class = torch.argmax(output, dim=1)
        predicted_numbers = (torch.topk(softmax, k=3, dim=1).indices + 1)
        # predicted_number = predicted_class.item() + 1  # 转换回1到16

    print(f"Predicted: {predicted_numbers.tolist()}\n")
    return predicted_numbers

def CalcMost(results):
    all_numbers = []
    for result in results:
        all_numbers.extend(result.tolist())
    from collections import Counter
    counter = Counter(all_numbers)
    # print (counter.most_common())
    for num, freq in counter.most_common():
        print(f"{num} \t[{freq}]")

if __name__ == "__main__":
    paramRed = LSTMRedBallHyperParameters()
    window_sizes = paramRed.window_sizes
    result_red = []
    for window_size in window_sizes:
        result_red.append(PredictRed(paramRed.epochs, window_size))

    param = LSTMHyperParameters()
    window_sizes = param.window_sizes
    result_blue = []
    for window_size in window_sizes:
        result_blue.append(Predict(param.epochs, window_size))
    CalcMost(result_blue)

