import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

# 设置海报级别的绘图风格 (字体大，线条粗)
plt.rcParams.update({'font.size': 14, 'axes.linewidth': 2, 'lines.linewidth': 3})

print("🎨 正在准备数据并绘制高清图...")

# 1. 数据加载与处理 (保持和之前一致)
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
if 'id' in df.columns:
    df = df.drop(columns=['id'])
df = df[df['gender'] != 'Other']

X = df.drop(columns=['stroke'])
y = df['stroke']

# 编码
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 训练 LightGBM (我们的主角)
model = LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# 图 1: 混淆矩阵 (Confusion Matrix)
# ==========================================
cm = confusion_matrix(y_test, y_pred)
# 计算百分比用于标注
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
# 使用蓝色系热力图
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Predicted Normal', 'Predicted Stroke'],
            yticklabels=['Actual Normal', 'Actual Stroke'],
            annot_kws={"size": 18, "weight": "bold"})

plt.title('Confusion Matrix: LightGBM', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig("poster_confusion_matrix.png", dpi=300)
print("✅ 混淆矩阵已保存: poster_confusion_matrix.png")

# ==========================================
# 图 2: ROC 曲线 (ROC Curve)
# ==========================================
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
# 画我们的曲线
plt.plot(fpr, tpr, color='#FF5733', label=f'LightGBM (AUC = {roc_auc:.2f})')
# 画基准线 (纯随机猜测)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guess (AUC = 0.50)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve Performance', fontsize=16, pad=20)
plt.legend(loc="lower right", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("poster_roc_curve.png", dpi=300)
print("✅ ROC 曲线已保存: poster_roc_curve.png")