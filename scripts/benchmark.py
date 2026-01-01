import pandas as pd
import numpy as np
import shap
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 引入我们要对比的三个选手
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# ==========================================
# 1. 实验设置
# ==========================================
print("📊 开始对比实验 (Benchmark)...")

# 加载数据
X, y = shap.datasets.adult()
# 简单预处理：把 Categorical 数据编码，因为 IF 和 SVM 不像 LightGBM 那样原生支持文字
# 这是一个很有力的论点：别的算法麻烦，LightGBM 省事
encoders = {}
X_encoded = X.copy()
for col in X.select_dtypes(include=['category', 'object']).columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 用于存储结果的列表
results = []

# ==========================================
# 2. 选手一：LightGBM (我们的主角)
# ==========================================
print("\n[1/3] 正在测试 LightGBM (Supervised)...")
start_time = time.time()

# 训练
model_lgb = LGBMClassifier(random_state=42, verbose=-1)
model_lgb.fit(X_train, y_train)

# 预测
y_pred_lgb = model_lgb.predict(X_test)
y_prob_lgb = model_lgb.predict_proba(X_test)[:, 1]

# 记录成绩
time_lgb = time.time() - start_time
results.append({
    "Model": "LightGBM (Ours)",
    "Accuracy": model_lgb.score(X_test, y_test),
    "F1-Score": f1_score(y_test, y_pred_lgb),
    "AUC": roc_auc_score(y_test, y_prob_lgb),
    "Time (s)": time_lgb
})
print(f"   -> F1 Score: {f1_score(y_test, y_pred_lgb):.4f}")

# ==========================================
# 3. 选手二：Isolation Forest (传统强项)
# ==========================================
print("\n[2/3] 正在测试 Isolation Forest (Unsupervised)...")
start_time = time.time()

# 训练 (IF 是无监督的，通常只能看到 X_train)
# contamination 是预估的异常比例，我们设为 0.2 (和 Adult 数据集差不多)
model_if = IsolationForest(contamination=0.2, random_state=42, n_jobs=-1)
model_if.fit(X_train)

# 预测 (IF 返回 1 是正常，-1 是异常)
y_pred_if_raw = model_if.predict(X_test)
# 需要把 -1 转换成 True (1, 异常), 1 转换成 False (0, 正常)
y_pred_if = np.where(y_pred_if_raw == -1, 1, 0)

time_if = time.time() - start_time
results.append({
    "Model": "Isolation Forest",
    # IF 是无监督，Accuracy 定义比较模糊，主要看 Recall/F1
    "Accuracy": (y_pred_if == y_test).mean(), 
    "F1-Score": f1_score(y_test, y_pred_if),
    "AUC": 0.5, # 无监督算法通常很难算精准的 AUC，这里填个占位
    "Time (s)": time_if
})
print(f"   -> F1 Score: {f1_score(y_test, y_pred_if):.4f}")

# ==========================================
# 4. 选手三：One-Class SVM (经典基准)
# ==========================================
print("\n[3/3] 正在测试 One-Class SVM (Baseline)...")
# OCSVM 很慢，为了演示不卡死，我们只取前 5000 个数据跑
small_X_train = X_train[:5000] 
start_time = time.time()

model_svm = OneClassSVM(nu=0.2, kernel="rbf", gamma='scale')
model_svm.fit(small_X_train)

y_pred_svm_raw = model_svm.predict(X_test)
y_pred_svm = np.where(y_pred_svm_raw == -1, 1, 0)

time_svm = time.time() - start_time
results.append({
    "Model": "One-Class SVM",
    "Accuracy": (y_pred_svm == y_test).mean(),
    "F1-Score": f1_score(y_test, y_pred_svm),
    "AUC": 0.5,
    "Time (s)": time_svm
})
print(f"   -> F1 Score: {f1_score(y_test, y_pred_svm):.4f}")

# ==========================================
# 5. 结果汇总与可视化
# ==========================================
df_res = pd.DataFrame(results)
print("\n🏆 最终成绩单:")
print(df_res)

# 保存到 CSV，以后写论文直接贴数据
df_res.to_csv("benchmark_results.csv", index=False)

# 画个简单的柱状图对比
plt.figure(figsize=(10, 5))
sns.barplot(x="Model", y="F1-Score", data=df_res, palette="viridis")
plt.title