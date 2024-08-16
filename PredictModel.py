import DataModel
import Model
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

def PredictBall(model, inputs):
    with torch.no_grad():
        output = model(inputs)
        predicted_numbers = (torch.topk(torch.softmax(output, dim=1), 3).indices + 1).reshape(-1)
    
    print(f"Predicted: {predicted_numbers.tolist()}\n")
    return predicted_numbers

def PredictSSQ():
    pass
