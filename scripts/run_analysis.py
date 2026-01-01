import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import joblib
import numpy as np
from data_loader import load_stroke_data # 确保这个文件存在

# 设置绘图风格，让字体更大更清晰，适合海报
plt.rcParams.update({'font.size': 14})

# ==========================================
# 0. 【核心配置开关】确保这里是 stroke！
# ==========================================
DATASET_NAME = "stroke"
print(f"🚀 当前模式: {DATASET_NAME} 数据集 - 准备生成SHAP图")

# 1. 设置路径
save_dir = r"D:\code\pythoncode\Anomaly Detection and Repair for Mixed Data Type Inputs"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ==========================================
# 2. 加载中风数据
# ==========================================
print("正在加载 Stroke (中风预测) 数据...")
X, y = load_stroke_data("healthcare-dataset-stroke-data.csv")

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. 训练 LightGBM 模型
# ==========================================
print("正在训练 LightGBM 模型 (用于SHAP分析)...")
# 使用 balanced 权重，这对于不平衡数据的SHAP分析更准确
model = lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
model.fit(X_train, y_train)

# ==========================================
# 4. 生成 SHAP 解释图 (核心修改部分)
# ==========================================
print("正在计算 SHAP 值...")
# 使用 TreeExplainer 解释模型
explainer = shap.TreeExplainer(model)
# 计算测试集的 SHAP 值
shap_values_all = explainer.shap_values(X_test)

# 【重要】LightGBM 二分类会返回两个数组的列表 [负类SHAP, 正类SHAP]
# 我们只关心导致中风(正类, label=1)的原因，所以取第二个数组 [1]
if isinstance(shap_values_all, list):
    shap_values_target = shap_values_all[1]
else:
    shap_values_target = shap_values_all

print("正在绘制高清 SHAP 蜂群图...")

# 创建一个大的画布，保证清晰度
plt.figure(figsize=(12, 8))

# 【关键】绘制 SHAP Summary Plot (蜂群图模式)
# 不加 plot_type="bar" 就会默认画出信息量更大的蜂群图
# 这种图不仅能看出谁重要，还能看出特征值高低对结果是正面还是负面影响
shap.summary_plot(shap_values_target, X_test, show=False)

# 保存为特定的高清文件用于海报
plot_filename = "poster_stroke_shap_summary.png"
plt.savefig(os.path.join(save_dir, plot_filename), bbox_inches='tight', dpi=400)

print("-" * 30)
print(f"✅ 中风数据集的专属 SHAP 图已保存为: {plot_filename}")
print("请检查文件夹，这张图比之前的条形图更专业！")
print("-" * 30)