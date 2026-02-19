import matplotlib.pyplot as plt
import numpy as np
from predatatool import read_txt
import statistics


results = read_txt('mag.txt', 3, 500)

average = statistics.mean(results[2])
print("平均值:", average)

categories = range(1,501)  # 类别（X轴）[i + 1 for i in range(500)]
values = results[2]         # 对应值（Y轴）

# 绘制柱状图
plt.bar(categories, values, color='skyblue', edgecolor='black')

# 添加标签和标题
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Simple Bar Chart')

# 显示图形
plt.show()