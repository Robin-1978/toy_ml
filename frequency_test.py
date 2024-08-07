from scipy.stats import chi2
import numpy as np

def frequency_test(sequence):
    n = len(sequence)
    observed_counts = np.bincount(sequence - 1, minlength=16)
    expected_count = n / 16
    
    chi2_stat = np.sum((observed_counts - expected_count) ** 2 / expected_count)
    p_value = 1 - chi2.cdf(chi2_stat, df=15)  # 自由度 = 类别数 - 1
    
    return chi2_stat, p_value, observed_counts, expected_count

# 示例用法
data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)
blue_ball = data[:, 8][::-1].astype(int)
chi2_stat, p_value, observed_counts, expected_count = frequency_test(blue_ball)
print(f"卡方统计量: {chi2_stat}")
print(f"p值: {p_value}")
print(f"观察到的频数: {observed_counts}")
print(f"期望的频数: {expected_count}")