import numpy as np
from scipy import stats
import DataModel
from scipy.stats import chisquare
from scipy.stats import kstest

def runs_test_simple(binary_sequence):

    # Ensure data is a numpy array
    data = np.asarray(binary_sequence)
    
    # Count the number of runs
    runs = np.sum(data[:-1] != data[1:]) + 1
    
    # Number of ones and zeros
    n1 = np.sum(data)
    n0 = len(data) - n1
    
    # Calculate the expected number of runs and variance
    expected_runs = 1 + (2 * n1 * n0) / (n1 + n0)
    variance_runs = (expected_runs - 1) * (expected_runs - 2) / (n1 + n0 - 1)
    
    # Test statistic
    test_statistic = (runs - expected_runs) / np.sqrt(variance_runs)
    
    # Calculate p-value using normal distribution
    p_value = 2 * (1 - stats.norm.cdf(np.abs(test_statistic)))
    
    return test_statistic, p_value

def runs_test(binary_sequence):
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
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))
    
    return z, p_value

def chi_square_test(data, num_bins):
    observed_counts, _ = np.histogram(data, bins=np.arange(1, num_bins + 2))
    expected_counts = len(data) / num_bins * np.ones(num_bins)
    chi2_stat, p_value = chisquare(observed_counts, expected_counts)
    return chi2_stat, p_value

def ks_test(data, num_bins):
    data_normalized = (data - 1) / (num_bins - 1)  # Normalizing to [0, 1]
    d_statistic, p_value = kstest(data_normalized, 'uniform')
    return d_statistic, p_value
    
def statics_test(binary_sequence):
    test_statistic, p_value = runs_test_simple(binary_sequence)
    print(f"Test Statistic: {test_statistic}")
    print(f"P-Value: {p_value}")
    
    # Interpret the result
    if p_value < 0.05:
        print("The sequence is not random.")
    else:
        print("The sequence appears to be random.")

if __name__ == '__main__':
    balls, diff = DataModel.load_ssq_blue_diff()

    # big little
    print('big little')
    binary_sequence = np.where(balls > 8, 1, 0)
    statics_test(binary_sequence)

    # diff
    print('diff')
    binary_sequence = np.where(np.diff(balls) > 0, 1, 0)
    statics_test(binary_sequence)

    # even odd
    print('even odd')
    binary_sequence = np.where(balls % 2 == 0, 1, 0)
    statics_test(binary_sequence)

    # mean
    print('mean')
    binary_sequence = np.where(balls > balls.mean(), 1, 0)
    statics_test(binary_sequence)

    observed_freq, _ = np.histogram(balls, bins=np.arange(0.5, 17.5, 1))
    expected_freq = len(balls) / 16 * np.ones(16)
    chi2_statistic, chi2_p_value = stats.chisquare(observed_freq, expected_freq)
    print(f'Chi-Square Statistic: {chi2_statistic}, P-Value: {chi2_p_value}')

    ks_statistic, ks_p_value = stats.kstest(balls, 'uniform', args=(1, 16))
    print(f'KS Statistic: {ks_statistic}, P-Value: {ks_p_value}')


    from statsmodels.tsa.stattools import adfuller
    import pandas as pd

    # 假设data为你的时间序列数据
    result = adfuller(balls)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    # 绘制ACF图和PACF图
    plot_acf(balls, lags=20)
    plot_pacf(balls, lags=20)
    plt.show()
    
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.figure(figsize=(12, 6))
    # sns.histplot(balls, bins=16, kde=True, color='blue', alpha=0.7)
    # plt.title('Distribution of Data')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()


    # from scipy.stats import uniform
    # data_sorted = np.sort(balls)
    # data_cdf = np.arange(1, len(balls) + 1) / len(balls)
    # uniform_cdf = uniform.cdf(data_sorted, loc=1, scale=15)

    # plt.figure(figsize=(12, 6))
    # plt.step(data_sorted, data_cdf, where='post', label='Empirical CDF', color='blue')
    # plt.plot(data_sorted, uniform_cdf, label='Uniform CDF', linestyle='--', color='red')
    # plt.title('CDF Comparison')
    # plt.xlabel('Value')
    # plt.ylabel('CDF')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
