import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import joblib

# 1. 设置路径
save_dir = r"D:\code\pythoncode"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 2. 准备数据
print("正在加载数据...")
X, y = shap.datasets.adult()
# 强制转换数据类型，避免后续 Gower 距离计算报错
# 将布尔列（True/False）转换为整数 0/1，将 Categorical 保持原样
for col in X.select_dtypes(include=['bool']).columns:
    X[col] = X[col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练模型
print("正在训练模型...")
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. 生成全局解释图 (Summary Plot)
print("正在生成解释图...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig(os.path.join(save_dir, "result_shap_importance.png"), bbox_inches='tight', dpi=300)

# ==========================================
# 【关键更新】保存模型和“正常数据库”
# ==========================================
print("正在保存系统文件...")

# 1. 保存模型
joblib.dump(model, os.path.join(save_dir, "model_lgb.pkl"))

# 2. 保存测试数据 (用于演示)
joblib.dump(X_test, os.path.join(save_dir, "test_data.pkl"))

# 3. 【新增】保存所有"正常"的训练数据
# 修复器需要在这个池子里找邻居。我们假设标签 0 (False) 是正常。
normal_data = X_train[y_train == 0]
joblib.dump(normal_data, os.path.join(save_dir, "normal_data.pkl"))

print("-" * 30)
print("✅ 准备工作完成！")
print(f"已保存 'normal_data.pkl' (包含 {len(normal_data)} 条正常样本)")
print("-" * 30)