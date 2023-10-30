import random

# 生成100个包含三个整数的数组
arrays = []
array_length = 1e8
group_count = 10
for _ in range(int(array_length)):
    array = [random.randint(0, 100), random.randint(1, group_count), random.randint(0, 100)]
    arrays.append(array)

# 将数组写入文件
with open(f'generateData/uniform_data_{array_length}_{group_count}.txt', 'w') as file:
    for array in arrays:
        line = ' '.join(map(str, array))  # 将整数列表转换为字符串
        file.write(line + '\n')
