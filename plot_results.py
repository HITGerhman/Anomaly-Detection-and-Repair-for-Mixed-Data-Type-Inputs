import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 手动填入你的跑分结果
data = {
    'Model': ['LightGBM (Ours)', 'Isolation Forest', 'One-Class SVM'],
    'F1-Score': [0.730690, 0.261261, 0.393750],
    'AUC': [0.9301, 0.5000, 0.5000]
}
df = pd.DataFrame(data)

# 2. 画图
plt.figure(figsize=(10, 6))

# 使用融化 (Melt) 技术以便在一个图里画两个指标
df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

# 画柱状图
sns.barplot(x="Model", y="Score", hue="Metric", data=df_melted, palette="viridis")

# 设置标题和标签
plt.title("Performance Comparison: Our Framework vs Baselines", fontsize=15)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.xlabel("")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱子上标数值
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.2f')

plt.tight_layout()
plt.savefig("final_benchmark_chart.png", dpi=300)
print("✅ 最终对比图已保存为 final_benchmark_chart.png")