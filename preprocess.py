import numpy as np

def ExtractSlidingWindow(sequence, window_size):
    """
    从序列中提取滑动窗口及其后一个数作为目标值

    Args:
        sequence: 输入序列
        window_size: 窗口大小，即子序列的长度

    Returns:
        X: 滑动窗口的数组，每行是一个长度为 window_size 的子序列
        y: 对应的目标值数组，每个元素是窗口后一个数
    """
    X = []
    y = []
    
    # 遍历序列，从索引 0 开始，直到可以形成长度为 window_size 的子序列
    for i in range(len(sequence) - window_size):
        # 提取当前位置开始的长度为 window_size 的子序列，并加入 X 列表
        window = sequence[i:i+window_size]
        X.append(window)
        
        # 将窗口的后一个数作为目标值 y
        y.append(sequence[i + window_size])
    
    # 将列表转换为 numpy 数组并返回
    return np.array(X), np.array(y)

#窗口均值
def MeanWindow(sequence):
    return np.mean(sequence, axis=1)

#窗口方差
def VarianceWindow(sequence):
    return np.var(sequence, axis=1)

#窗口标准差
def StandardDeviationWindow(sequence):
    return np.std(sequence, axis=1)

#窗口最大值
def MaxWindow(sequence):
    return np.max(sequence, axis=1)

#窗口最小值
def MinWindow(sequence):
    return np.min(sequence, axis=1)

#窗口数字线性回归系数
def LinearRegressionWindow(sequence):
    X = []
    for window in sequence:
        X.append( np.polyfit(np.arange(len(window)), window, 1))
    return np.array(X)

#窗口数字频率特征
def FrequencyWindow(sequence):
    return np.array([np.count_nonzero(sequence == i) for i in np.unique(sequence)])

#计算相邻数字之间的差值，并统计差值的特征，如差值的均值、标准差等。
def DifferenceWindow(sequence):
    return np.std(np.diff(sequence))