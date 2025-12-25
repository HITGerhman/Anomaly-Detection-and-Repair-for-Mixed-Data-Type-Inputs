import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import joblib  # 【新增】用来保存模型和数据的库

# 1. 设置路径
save_dir = r"D:\code\pythoncode"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 2. 准备数据
print("正在加载数据...")
X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练模型
print("正在训练模型...")
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. 生成 SHAP 图
print("正在生成解释图...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig(os.path.join(save_dir, "result_shap_importance.png"), bbox_inches='tight', dpi=300)

# ==========================================
# 【新增的核心步骤】保存模型和测试数据
# ==========================================
print("正在保存模型文件...")
# 保存模型 (model.pkl)
joblib.dump(model, os.path.join(save_dir, "model_lgb.pkl"))
# 保存测试数据 (test_data.pkl) - 网页演示时需要用到这些数据
joblib.dump(X_test, os.path.join(save_dir, "test_data.pkl"))

print("-" * 30)
print("✅ 准备工作完成！")
print(f"1. 模型文件已保存: model_lgb.pkl")
print(f"2. 数据文件已保存: test_data.pkl")
print(f"3. 结果图片已保存: result_shap_importance.png")
print("-" * 30)