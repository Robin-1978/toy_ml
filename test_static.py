import numpy as np
import torch
import torch.nn as nn

def one_hot_encode(ball_numbers, num_classes=33):
    # ball_numbers: shape (batch_size, window_size, input_size)
    batch_size, window_size, input_size = ball_numbers.shape
    one_hot_encoded = np.zeros((batch_size, window_size, input_size, num_classes))
    
    for b in range(batch_size):
        for w in range(window_size):
            for i in range(input_size):
                one_hot_encoded[b, w, i, ball_numbers[b, w, i]] = 1
                
    return one_hot_encoded

# Example usage
ball_numbers = np.random.randint(0, 33, size=(10, 5, 6))  # (batch_size, window_size, input_size)
one_hot_encoded = torch.tensor(one_hot_encode(ball_numbers), dtype=torch.float)

batch_size, window_size, input_size, num_classes = one_hot_encoded.shape
conv1 = nn.Conv2d(in_channels=33, out_channels=64, kernel_size=(1, 1))
x = one_hot_encoded.permute(0, 3, 1, 2)
x = conv1(x)


pool = nn.MaxPool2d(kernel_size=(1, 2))
x = pool(x)

global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
x = global_avg_pool(x)

conv_output_size = 64 * (input_size // 2)
x = x.reshape(batch_size, window_size, conv_output_size)  # (batch_size, window_size, 64 * (input_size // 2))

print (x)