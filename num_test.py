import math
import numpy as np
from scipy.stats import chisquare, entropy, norm, chi2

def frequency_test(random_numbers):
    #频率检验
    values, counts = np.unique(random_numbers, return_counts=True)

    # 使用卡方检验检查分布均匀性
    chi2_stat, p_value = chisquare(counts)

    print(f"Chi2 Stat: {chi2_stat}")
    print(f"P-Value: {p_value}")

    if p_value < 0.05:
        print("随机数可能不是均匀分布的。")
    else:
        print("随机数看起来是均匀分布的。")


def serial_test(random_numbers):
    # 序列检验
    pass

def poker_test(random_numbers, m=4):
    n = len(random_numbers)
    
    # 将随机数分成长度为m的组
    num_groups = n // m
    groups = np.array_split(random_numbers[:num_groups * m], num_groups)
    
    # 计算每组中唯一数字的个数
    unique_counts = np.array([len(np.unique(group)) for group in groups])
    
    # 计算每种可能的唯一数字个数出现的次数
    observed_counts = np.array([np.sum(unique_counts == k) for k in range(1, m + 1)])
    
    # 计算期望出现次数
    # expected_counts = np.array([
    #     num_groups * (math.comb(m, k) * (10**k * (10-1)**(m-k)) * (1/10)**m)
    #     for k in range(1, m + 1)
    # ])
    
    # 计算卡方统计量
    chi2_stat = np.sum((observed_counts - expected_counts)**2 / expected_counts)
    
    # 计算自由度
    degrees_of_freedom = m - 1
    
    # 计算p值
    p_value = 1 - chi2.cdf(chi2_stat, degrees_of_freedom)
    
    return chi2_stat, p_value, observed_counts, expected_counts


def runs_test(random_numbers):
    n = len(random_numbers)
    
    # 将随机数转换为二进制序列
    median = np.median(random_numbers)
    binary_sequence = (random_numbers > median).astype(int)
    
    # 计算0和1的个数
    n1 = np.sum(binary_sequence)
    n0 = n - n1

    # 计算runs的数量
    runs = 1
    for i in range(1, n):
        if binary_sequence[i] != binary_sequence[i - 1]:
            runs += 1

    # 计算期望runs的数量和方差
    expected_runs = (2 * n1 * n0) / n + 1
    if n1 == 0 or n0 == 0:
        variance_runs = 0
    else:
        variance_runs = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1))

    # 确保方差非负
    if variance_runs < 0:
        variance_runs = 0

    # 计算z值
    if variance_runs == 0:
        z = 0
    else:
        z = (runs - expected_runs) / np.sqrt(variance_runs)

    # 计算p值
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return runs, expected_runs, variance_runs, z, p_value


def autocorrelation_test(random_numbers):
    #自相关检验
    pass

def entropy_evaluation(random_numbers):
    #熵评估
    pass

def spectral_test(random_numbers):
    # 谱检验
    pass

# 从CSV文件加载数据，假设数据格式为 id,date,red_ball1,red_ball2,red_ball3,red_ball4,red_ball5,red_ball6,blue_ball
data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)

# 获取蓝球号码数据，并逆序排列
blue_ball = data[:, 8]
blue_ball = blue_ball[::-1].astype(int)

frequency_test(blue_ball)

runs, expected_runs, variance_runs, z, p_value = runs_test(blue_ball)

print(f"随机数序列: {blue_ball}")
print(f"实际runs数量: {runs}")
print(f"期望runs数量: {expected_runs}")
print(f"方差: {variance_runs}")
print(f"z值: {z}")
print(f"p值: {p_value}")

if p_value < 0.05:
    print("拒绝原假设，序列可能不是独立的。")
else:
    print("未拒绝原假设，序列看起来是独立的。")

chi2_stat, p_value, observed_counts, expected_counts = poker_test(blue_ball)

print(f"卡方统计量: {chi2_stat}")
print(f"p值: {p_value}")
print(f"观察到的次数: {observed_counts}")
print(f"期望的次数: {expected_counts}")

if p_value < 0.05:
    print("拒绝原假设，序列可能不是独立的。")
else:
    print("未拒绝原假设，序列看起来是独立的。")
