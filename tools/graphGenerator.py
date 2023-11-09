import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 输入数据
x = [0, 0.25 ,0.5 ,0.75 ,1]
y1 = [1.38306, 34.335, 67.3589, 100.371, 133.389] #built-in triangle RT
y2 = [4.30103 ,22.8589 , 41.3972, 62.6812, 80.6169]  #cube RT
y3 = [33426.1, 40273.8, 45974.3, 50965.8, 57638.7]  #mysql default
y4 = [2971.91, 40408.1, 46095.4, 50955.9, 56215.3]  #mysql index
y5 = [1195.36, 1734.23, 2255.95, 2597.27, 2981.59]  #pgsql default
y6 = [1448.67, 1736.82, 2260.55, 2607.08, 2997]  #pgsql index
y7 = [586.901, 816.091, 1037.64, 1197.53, 1371.05]  #pgsql multi-core

# 设置颜色代码
color1 = "#038355" # 孔雀绿
color2 = "#ffc34e" # 向日黄
color3 = "#ae5e52" # 辰砂
color4 = "#6b9ac8" # 竹月
color5 = "#f8c6a9" # 十祥棉
color6 = "#c9e2e3" # 云灰
color7 = "#c85454" # 银朱

# 设置字体
font = {'family' : 'Times New Roman',
        'size'   : 12}
plt.rc('font', **font)

# 绘图
sns.set_style("whitegrid") # 设置背景样式
sns.lineplot(x=x, y=y1, color=color1, linewidth=2.0, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='RT built-in')
sns.lineplot(x=x, y=y2, color=color2, linewidth=2.0, marker="s", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='RT cube')
#sns.lineplot(x=x, y=y3, color=color3, linewidth=2.0, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='mysql default')
#sns.lineplot(x=x, y=y4, color=color4, linewidth=2.0, marker="s", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='mysql using idx')
#sns.lineplot(x=x, y=y5, color=color5, linewidth=2.0, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='pgsql default')
#sns.lineplot(x=x, y=y6, color=color6, linewidth=2.0, marker="s", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='pgsql using idx')
#sns.lineplot(x=x, y=y7, color=color7, linewidth=2.0, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5, label='pgsql multi-core')

# 添加标题和标签
plt.title("Title", fontweight='bold', fontsize=14)
plt.xlabel("Selectivity", fontsize=12)
plt.ylabel("Time / ms", fontsize=12)

# 添加图例
plt.legend(loc='upper left', frameon=True, fontsize=6)

# 设置刻度字体和范围
x_ticks = np.linspace(0, 1, 5)
plt.xticks(x_ticks)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlim(0, 1)
#plt.ylim(0, 25)

# 设置坐标轴样式
for spine in plt.gca().spines.values():
    spine.set_edgecolor("#CCCCCC")
    spine.set_linewidth(1.5)

plt.savefig('lineplot.png', dpi=300, bbox_inches='tight')