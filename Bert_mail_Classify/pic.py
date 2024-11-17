import matplotlib.pyplot as plt
import numpy as np

# 分类报告数据
categories = ['Class 0', 'Class 1']
precision = [0.99, 0.98]
recall = [1.00, 0.97]
f1_score = [1.00, 0.98]

# 设置柱状图的位置
x = np.arange(len(categories))
width = 0.2  # 每个柱状图的宽度

fig, ax = plt.subplots(figsize=(8, 5))
bar1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
bar2 = ax.bar(x, recall, width, label='Recall', color='salmon')
bar3 = ax.bar(x + width, f1_score, width, label='F1 Score', color='lightgreen')

# 添加标签和标题
ax.set_xlabel('Categories')
ax.set_ylabel('Scores')
ax.set_title('Classification Report Metrics by Category')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# 显示数值
for bars in [bar1, bar2, bar3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',  # 显示到小数点后两位
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 标签的垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()

