import torch
import DataModel
import LSTMModel
from config import LSTMHyperParameters

def PrepareData():
    table = DataModel.LoadData("data/ssq.db")
    data = table["Ball_7"].values
    return data

def PrepareWindow(seq_data, window_size):
    inputs = torch.tensor(seq_data[-window_size:], dtype=torch.long).unsqueeze(0)  # Shape: (1, window_size, num_classes)
    # print (inputs)
    return inputs

def Predict(epoch_num, window_size):
    param = LSTMHyperParameters()
    # torch.save(model.state_dict(), f"LSTMModel_{window_size}.pth")
    model = LSTMModel.LSTMModel(input_size=param.input_size, hidden_size=param.hidden_size, num_classes=param.num_classes, num_layers=param.num_layers, dropout=param.dropout)
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
    param = LSTMHyperParameters()
    window_sizes = param.window_sizes
    result = []
    for window_size in window_sizes:
        result.append(Predict(param.epochs, window_size))
    CalcMost(result)
