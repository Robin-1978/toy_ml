import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import chi2

def autocorrelation_test(sequence, lags=10):
    """
    自动相关检验，用于检验序列的随机性

    Args:
        sequence: 待检验的序列
        lags: 滞后阶数

    Returns:
        acf_values: 自动相关系数
        p_value: p值
    """
    acf_values = acf(sequence, nlags=lags)
    Q = len(sequence) * np.sum(np.square(acf_values[1:]))
    p_value = 1 - chi2.cdf(Q, lags)
    return acf_values, Q, p_value

# 示例用法
data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)
blue_ball = data[:, 8][::-1].astype(int)
acf_values, Q, p_value = autocorrelation_test(blue_ball, lags=10)
print(f"自动相关系数: {acf_values}")
print(f"Q值: {Q}")
print(f"p值: {p_value}")

if p_value < 0.05:
    print("拒绝原假设，序列可能不是独立的。")
else:
    print("未拒绝原假设，序列看起来是独立的。")