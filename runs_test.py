import numpy as np
from scipy.stats import norm
import math
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def calculate_runs_stats(n1, n0, n):
    """
    计算游程检验所需的期望值和方差

    Args:
        n1: 上升的次数
        n0: 下降的次数
        n: 序列长度

    Returns:
        tuple: 期望运行次数和方差
    """
    print(f"n0: {n0} \tn1: {n1} \tn: {n}")
    expected_runs = (2 * n1 * n0) / n + 1

    # 使用对数运算避免溢出
    log_numerator = math.log(2 * n1 * n0) + math.log(2 * n1 * n0 - n)
    log_denominator = 2 * math.log(n) + math.log(n - 1)
    variance_runs = math.exp(log_numerator - log_denominator)

    # 设置最小方差
    variance_runs = max(variance_runs, 1e-10)

    return expected_runs, variance_runs

def runs_test(sequence):
    """
    游程检验，用于检验序列的随机性

    Args:
        sequence: 待检验的序列

    Returns:
        tuple: 实际运行次数、期望运行次数、方差、z值、p值
    """
    # 检查输入序列
    if not isinstance(sequence, np.ndarray):
        raise TypeError("输入序列必须为NumPy数组")
    if len(sequence) < 2:
        raise ValueError("序列长度至少为2")

    # 将序列转换为二进制表示
    binary_sequence = np.where(np.diff(sequence) > 0, 1, 0)

    # 计算运行次数
    num_runs = np.sum(np.concatenate(([1], binary_sequence != np.roll(binary_sequence, 1), [1])))

    # 计算n1和n0
    n1 = np.sum(binary_sequence)
    n0 = len(binary_sequence) - n1

    # 计算期望值和方差
    expected_runs, variance_runs = calculate_runs_stats(n1, n0, len(binary_sequence))

    # 计算z值和p值
    z = (num_runs - expected_runs) / np.sqrt(variance_runs)
    p_value = 2 * (1 - norm.cdf(np.abs(z)))

    return num_runs, expected_runs, variance_runs, z, p_value


# def runs_org_test(sequence):
#     if not isinstance(sequence, np.ndarray):
#         raise TypeError("Input sequence must be a NumPy array")
#     if len(sequence) < 2:
#         raise ValueError("Sequence length must be at least 2")

#     binary_sequence = np.where(np.diff(sequence) > 0, 1, 0)
#     n1 = np.sum(binary_sequence)
#     n2 = len(binary_sequence) - n1

#     if n1 == 0 or n2 == 0:
#         runs_exp = 0
#         stan_dev = 0
#     else:
#         runs_exp = ((2 * n1 * n2) / (n1 + n2)) + 1
#         stan_dev = math.sqrt(
#             (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
#         )

#     runs = np.sum(np.concatenate(([1], binary_sequence != np.roll(binary_sequence, 1), [1])))
#     z = (runs - runs_exp) / stan_dev
#     p_value = 2 * (1 - norm.cdf(np.abs(z))) if stan_dev != 0 else 0

#     return runs, runs_exp, stan_dev * stan_dev, z, p_value

def runs_org_test(binary_sequence):
    # 计算实际运行数量
    runs = np.sum(np.concatenate(([1], binary_sequence != np.roll(binary_sequence, 1), [1])))
    
    # 计算期望运行数量
    n1 = np.sum(binary_sequence)
    n0 = len(binary_sequence) - n1
    expected_runs = (2 * n1 * n0 / len(binary_sequence)) + 1
    
    # 计算方差
    variance_runs = (2 * n1 * n0 * (2 * n1 * n0 - len(binary_sequence))) / (len(binary_sequence)**2 * (len(binary_sequence) - 1))
    
    # 计算z值
    z = (runs - expected_runs) / np.sqrt(variance_runs)
    
    # 计算p值
    p_value = 2 * (1 - norm.cdf(np.abs(z)))
    
    return runs, expected_runs, variance_runs, z, p_value


# 加载数据
data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)
blue_ball = data[:, 8][::-1].astype(int)

# blue_ball = np.array([15,13,8,7,6,16,3,16,12,11])

# 执行游程检验
# runs, expected_runs, variance_runs, z, p_value = runs_test(blue_ball)
# print(f"实际运行数量: {runs}")
# print(f"期望运行数量: {expected_runs}")
# print(f"方差: {variance_runs}")
# print(f"z值: {z}")
# print(f"p值: {p_value}")

# if p_value < 0.05:
#     print("拒绝原假设，序列可能不是独立的。")
# else:
#     print("未拒绝原假设，序列看起来是独立的。")


print("升降序检验")
binary_sequence = np.where(np.diff(blue_ball) > 0, 1, 0)
runs, expected_runs, variance_runs, z, p_value = runs_org_test(binary_sequence)
print(f"实际运行数量: {runs}")
print(f"期望运行数量: {expected_runs}")
print(f"方差: {variance_runs}")
print(f"z值: {z}")
print(f"p值: {p_value}")
if p_value < 0.05:
    print("\033[31m拒绝原假设, 序列可能不是独立的。\033[0m")
else:
    print("\033[32m未拒绝原假设, 序列看起来是独立的。\033[0m")

print("大小数检验")
binary_sequence = np.where(blue_ball > 8, 1, 0)
runs, expected_runs, variance_runs, z, p_value = runs_org_test(binary_sequence)
print(f"实际运行数量: {runs}")
print(f"期望运行数量: {expected_runs}")
print(f"方差: {variance_runs}")
print(f"z值: {z}")
print(f"p值: {p_value}")
if p_value < 0.05:
    print("\033[31m拒绝原假设, 序列可能不是独立的。\033[0m")
else:
    print("\033[32m未拒绝原假设, 序列看起来是独立的。\033[0m")


print("奇偶数检验")
binary_sequence = np.where(blue_ball % 2 == 1, 1, 0)
runs, expected_runs, variance_runs, z, p_value = runs_org_test(binary_sequence)
print(f"实际运行数量: {runs}")
print(f"期望运行数量: {expected_runs}")
print(f"方差: {variance_runs}")
print(f"z值: {z}")
print(f"p值: {p_value}")
if p_value < 0.05:
    print("\033[31m拒绝原假设, 序列可能不是独立的。\033[0m")
else:
    print("\033[32m未拒绝原假设, 序列看起来是独立的。\033[0m")
# # 执行Ljung-Box检验
# result = sm.stats.acorr_ljungbox(blue_ball, lags=[1, 5, 10, 20])
# print(f"Ljung-Box检验结果 (多阶滞后):\n{result}")


# adf_result = adfuller(blue_ball)
# print('ADF Statistic: %f' % adf_result[0])
# print('p-value: %f' % adf_result[1])
# print('Critical Values:')
# for key, value in adf_result[4].items():
#     print('\t%s: %.3f' % (key, value))

# if adf_result[1] < 0.05:
#     print("拒绝原假设，序列没有单位根，可能是平稳的。")
# else:
#     print("未拒绝原假设，序列可能有单位根，非平稳。")



#     # 自相关图
# fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# # 序列图
# ax[0].plot(blue_ball)
# ax[0].set_title("Blue Ball Sequence Plot")
# ax[0].set_xlabel("Index")
# ax[0].set_ylabel("Value")

# # 自相关图
# sm.graphics.tsa.plot_acf(blue_ball, lags=162, ax=ax[1])
# ax[1].set_title("Autocorrelation Function (ACF)")
# plt.show()
