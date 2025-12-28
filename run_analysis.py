import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import joblib
from data_loader import load_stroke_data # 确保你已经写好了这个文件

# ==========================================
# 0. 【核心配置开关】在这里切换数据集！
# ==========================================
# 选项: "adult" 或 "stroke"
DATASET_NAME = "stroke"  # <--- 想跑哪个，改这里就行！

print(f"🚀 当前模式: {DATASET_NAME} 数据集")

# 1. 设置路径
save_dir = r"D:\code\pythoncode"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ==========================================
# 2. 智能数据加载
# ==========================================
if DATASET_NAME == "adult":
    print("正在加载 Adult (人口普查) 数据...")
    X, y = shap.datasets.adult()
    # 简单的预处理
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)

elif DATASET_NAME == "stroke":
    print("正在加载 Stroke (中风预测) 数据...")
    # 调用我们写的加载器
    X, y = load_stroke_data("healthcare-dataset-stroke-data.csv")
    
else:
    raise ValueError("不支持的数据集名称！请使用 'adult' 或 'stroke'")

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. 训练模型 (通用逻辑，不需要改)
# ==========================================
print("正在训练 LightGBM 模型...")
model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. 生成解释图
# ==========================================
print("正在生成 SHAP 解释...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig(os.path.join(save_dir, "result_shap_importance.png"), bbox_inches='tight', dpi=300)

# ==========================================
# 5. 保存结果与元数据
# ==========================================
print("正在保存系统文件...")

joblib.dump(model, os.path.join(save_dir, "model_lgb.pkl"))
joblib.dump(X_test, os.path.join(save_dir, "test_data.pkl"))

# 保存正常样本 (用于修复模块)
# Adult: y==False(0) 是正常; Stroke: y==0 是正常
normal_data = X_train[y_train == 0]
joblib.dump(normal_data, os.path.join(save_dir, "normal_data.pkl"))

# 【关键】把当前用的是哪个数据集也存下来！
# 这样网页端(app.py)就知道该显示什么标题了
config_data = {
    "dataset_name": DATASET_NAME,
    "feature_names": list(X.columns)
}
joblib.dump(config_data, os.path.join(save_dir, "config.pkl"))

print("-" * 30)
print(f"✅ 完成！已保存 {DATASET_NAME} 模式的所有文件。")
print("-" * 30)