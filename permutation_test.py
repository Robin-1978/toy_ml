import numpy as np

def permutation_test(sequence, num_permutations=1000):
    """
    重排检验，用于检验序列的随机性

    Args:
        sequence: 待检验的序列
        num_permutations: 重排次数

    Returns:
        p_value: p值
    """
    original_mean = np.mean(sequence)
    permuted_means = np.zeros(num_permutations)

    for i in range(num_permutations):
        permuted_sequence = np.random.permutation(sequence)
        permuted_means[i] = np.mean(permuted_sequence)

    p_value = np.sum(np.abs(permuted_means - original_mean) >= np.abs(original_mean - np.mean(sequence))) / num_permutations
    return p_value

# 示例用法
data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)
blue_ball = data[:, 8][::-1].astype(int)
p_value = permutation_test(blue_ball)
print(f"p值: {p_value}")

if p_value < 0.05:
    print("拒绝原假设，序列可能不是独立的。")
else:
    print("未拒绝原假设，序列看起来是独立的。")
