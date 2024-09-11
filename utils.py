import torch
import numpy
import random
import datetime


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log(message):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}")


def PrepareData(df, features=[], targets=[], window_size=5):
    X = []
    y = []
    for i in range(len(df) - window_size):
        X.append(df[features].iloc[i : i + window_size].values)
        y.append(df[targets].iloc[i + window_size])
    return (
        numpy.array(X),
        numpy.array(y),
        numpy.expand_dims(numpy.array(df[features].iloc[-window_size:]), axis=0),
    )
