import numpy as np
import random

# 生成随机整数的函数
def generate_random_numbers():
    number1 = random.randint(0, 100)
    number2 = random.randint(1, 10)
    number3 = random.randint(0, 100)
    return number1, number2, number3

# 创建长度为 1e8 的数组
array_length = int(1e2)
data = np.empty((array_length, 3), dtype=int)

# 填充数组
for i in range(array_length):
    data[i] = generate_random_numbers()

# 将数组保存到文件
np.save('random_data.npy', data)
