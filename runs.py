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
    binary_sequence = np.where(balls > 8, 1, 0)
    statics_test(binary_sequence)

    # diff
    binary_sequence = np.where(np.diff(balls) > 0, 1, 0)
    statics_test(binary_sequence)

    # even odd
    binary_sequence = np.where(balls % 2 == 0, 1, 0)
    statics_test(binary_sequence)

    chi2_stat, p_value = chi_square_test(balls, 16)
    print(f"Chi-Square Statistic: {chi2_stat}")
    print(f"P-Value: {p_value}")

    d_statistic, p_value = ks_test(balls, 16)
    print(f"KS Statistic: {d_statistic}")
    print(f"P-Value: {p_value}")