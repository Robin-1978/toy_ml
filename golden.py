import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt

# Step 1: 获取黄金价格数据
gold_data = yf.Ticker("GLD")
hist = gold_data.history(period="5y")  # 获取5年内的价格数据
hist.to_csv('gold_price_data.csv')

# 仅使用 "Close" 列 (收盘价)
data = hist[['Close']].dropna()


# 展示
# plt.plot(data)
# plt.show()