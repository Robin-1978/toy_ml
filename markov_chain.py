import numpy as np
np.random.seed(42)
# 从CSV文件加载数据，假设数据格式为 id,date,red_ball1,red_ball2,red_ball3,red_ball4,red_ball5,red_ball6,blue_ball
data = np.loadtxt('./data/ssq/data.csv', delimiter=',', skiprows=1)

# 获取蓝球号码数据，并逆序排列
blue_ball = data[:, 8]
blue_ball = blue_ball[::-1].astype(int)

# 状态空间为篮球的数字（1到16）
states = np.arange(1, 17)

# 初始化状态转移计数矩阵
transition_count = np.zeros((16, 16))

# 统计每个状态的出现次数
# state_counts = np.zeros(16)

# 计算篮球号码之间的转移次数和每个状态的出现次数
for i in range(len(blue_ball) - 1):
    current_ball = int(blue_ball[i])
    next_ball = int(blue_ball[i + 1])
    transition_count[current_ball - 1, next_ball - 1] += 1
    # state_counts[current_ball - 1] += 1

# 计算状态转移概率矩阵
# transition_matrix = transition_count / state_counts[:, np.newaxis]

# 计算状态转移概率矩阵
transition_matrix = transition_count / np.sum(transition_count, axis=1, keepdims=True)

# # 定义状态转移概率矩阵
# # 计算篮球号码之间的转移次数
# for i in range(len(blue_ball) - 1):
#     current_ball = int(blue_ball[i])
#     next_ball = int(blue_ball[i + 1])
#     transition_count[current_ball - 1, next_ball - 1] += 1

# # 计算状态转移概率矩阵
# transition_matrix = transition_count / np.sum(transition_count, axis=1, keepdims=True)

# 打印转移概率矩阵
print("转移概率矩阵:")
print(transition_matrix)
# 定义初始状态和初始概率分布（假设初始状态为蓝球数据的第一个号码）
initial_state = int(blue_ball[-1])


# 生成马尔可夫链序列的函数
def generate_markov_chain(initial_state, transition_matrix, steps=10):
    current_state = initial_state
    chain = [current_state]
    
    for _ in range(steps):
        # 获取当前状态对应的概率分布向量
        prob_vector = transition_matrix[current_state - 1]
        # 找到具有最大概率的下一个状态的索引
        next_state_index = np.argmax(prob_vector)
        # 将索引转换为实际状态值
        next_state = states[next_state_index]
        # 添加到链中
        chain.append(next_state)
        # 更新当前状态
        current_state = next_state
    
    return chain

# 生成一个长度为10的马尔可夫链序列
markov_chain = generate_markov_chain(initial_state, transition_matrix, steps=1)
print("生成的马尔可夫链序列:", markov_chain)


def build_higher_order_transition_matrix(data, order=2):
  """
  构建高阶马尔可夫链转移矩阵

  Args:
      data: 数据序列
      order: 马尔科夫链的阶数

  Returns:
      高阶转移概率矩阵
  """

  num_states = len(set(data))
  transition_count = np.zeros((num_states,) * (order + 1))

  for i in range(order, len(data)):
      index_tuple = tuple(int(x) - 1 for x in data[i-order:i+1])
      transition_count[index_tuple] += 1

  # Check for rows with zero sum (Solution 1)
  row_sums = np.sum(transition_count, axis=-1)
  zero_sum_rows = np.where(row_sums == 0)[0]

  # Handle zero counts (e.g., Laplace smoothing)
#   if len(zero_sum_rows) > 0:
#       transition_count += 1  # Add a small value to all counts

  # Normalize (avoid division by zero)
  transition_matrix = transition_count / np.sum(transition_count, axis=-1, keepdims=True)

  return transition_matrix

def generate_high_order_markov_chain(initial_states, transition_matrix, steps=10):
    """
    生成高阶马尔可夫链序列

    Args:
        data: 数据序列（用于获取状态集合）
        initial_states: 初始状态序列
        transition_matrix: 转移概率矩阵
        steps: 生成序列的长度

    Returns:
        生成的马尔可夫链序列
    """

    states = list(range(1, 17))  # 定义状态空间

    # 检查初始状态是否有效
    for state in initial_states:
        if state not in states:
            raise ValueError("初始状态不在状态空间内")

    chain = list(initial_states)
    for _ in range(steps):
        current_state_tuple = tuple(chain[-transition_matrix.ndim:])
        prob_vector = transition_matrix[current_state_tuple]

        try:
            max_prob_index = np.argmax(prob_vector)
            next_state = states[max_prob_index]
        except IndexError as e:
            print(f"Error: Index out of bounds. prob_vector: {prob_vector}")
            next_state = np.random.choice(states)

        chain.append(next_state)

    return chain

order = 1
# 构建二阶马尔科夫链转移矩阵
transition_matrix_2 = build_higher_order_transition_matrix(blue_ball, order)
print("二阶转移概率矩阵:", transition_matrix_2)
# 生成二阶马尔科夫链序列
#获取最后三个号码的前两个号码
initial_states = blue_ball[-order:]
markov_chain = generate_high_order_markov_chain(initial_states, transition_matrix_2, steps=1)
print("生成的马尔科夫链序列:", markov_chain)