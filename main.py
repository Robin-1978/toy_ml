import torch
import numpy as np
import torch.nn as nn

import requests
from bs4 import BeautifulSoup

def L_3D():

    # 假设原始数据是一个numpy数组
    data = np.array([[1, 2, 3], [4, 5, 6]])

    # 获取号码的种类数
    num_classes = 10

    # 将数据转换为Tensor
    data_tensor = torch.tensor(data)

    # 使用torch.nn.functional.one_hot进行编码
    one_hot_data = torch.nn.functional.one_hot(data_tensor, num_classes=num_classes)
    print(one_hot_data)


def L_6Plus1():
    red_data = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

    # 
    blue_data = np.array([[1], [2]])

    # Red One-hot编码
    num_red_classes = 35
    red_tensor = torch.tensor(red_data)
    red_one_hot = torch.nn.functional.one_hot(
        red_tensor - 1, num_classes=num_red_classes
    )  # 注意：索引从0开始，所以减1
    print("Red")
    print(red_one_hot)
    # Blue
    num_blue_classes = 12
    blue_tensor = torch.tensor(blue_data)
    blue_one_hot = torch.nn.functional.one_hot(
        blue_tensor - 1, num_classes=num_blue_classes
    )
    # Pad zeros to match red_one_hot size in the concatenating dimension
    padded_blue_one_hot = torch.nn.functional.pad(blue_one_hot, (0, num_red_classes - num_blue_classes, 0, 0))
    print("Blue")
    print(padded_blue_one_hot)

    # Now concatenate
    one_hot_data = torch.cat((red_one_hot, padded_blue_one_hot), dim=1)
    print("Concatenated")
    print(one_hot_data)

def L_6Plus2():
    red_data = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

    blue_data = np.array([[1], [2]])


    # 创建嵌入层
    num_red_classes = 35
    num_blue_classes = 12
    embedding_dim = 10

    embedding_red = nn.Embedding(num_red_classes, embedding_dim)
    embedding_blue = nn.Embedding(num_blue_classes, embedding_dim)

    # 获取嵌入向量
    red_embeddings = embedding_red(torch.nn.functional.one_hot(torch.tensor(red_data)))
    print(red_embeddings)
    blue_embeddings = embedding_blue(torch.nn.functional.one_hot(torch.tensor(blue_data)))
    print(blue_embeddings)

    # 拼接嵌入向量
    embeddings = torch.cat([red_embeddings, blue_embeddings], dim=1)  # 注意维度
    print(embeddings)

    print 

def get_ssq_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # 根据网页结构提取数据，例如：
    data = []
    for item in soup.select('.ssq-item'):
        # 解析每个开奖项目的数据
        date = item.select_one('.date').text
        red_balls = [int(ball.text) for ball in item.select('.red-ball')]
        blue_ball = int(item.select_one('.blue-ball').text)
        data.append((date, red_balls, blue_ball))
    return data



# 替换成实际的彩票网站URL
url = 'https://www.caipiao.com.cn/ssq/history/'
ssq_data = get_ssq_data(url)
print(ssq_data)

