import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from data_loader import load_stroke_data # 保持你的数据加载

# ==========================================
# 1. 设置海报级绘图风格 (加大加粗)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.linewidth': 2,
    'legend.fontsize': 16,
    'figure.figsize': (14, 8)
})

# ==========================================
# 2. 数据准备
# ==========================================
print("正在加载数据...")
# 加载原始数据
X_raw, y = load_stroke_data("healthcare-dataset-stroke-data.csv")

# 【重要预处理】为了让所有模型(包括RF和LR)都能跑，我们需要把分类变量转为数字
# LightGBM其实不需要这一步，但为了公平对比和代码不报错，我们统一做 One-Hot 编码
print("正在进行特征编码 (One-Hot Encoding)...")
X_encoded = pd.get_dummies(X_raw, drop_first=True)

# 某些模型(如LR)对数值幅度敏感，建议标准化 (虽然树模型不需要，但加了也没坏处)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========================================
# 3. 定义四大金刚模型
# ==========================================
# 注意：所有有监督模型都开启 class_weight='balanced' 以应对不平衡
models = {
    # --- 你的主角 ---
    # 修改 LightGBM 模型定义，采用“小步慢跑 + 限制复杂度”策略
'LightGBM (Tuned)': lgb.LGBMClassifier(
    random_state=42, 
    class_weight='balanced', 
    verbose=-1,
    
    # --- 关键调参区域 ---
    n_estimators=500,     # 增加树的数量 (原本100)
    learning_rate=0.02,   # 降低学习率，学得更细 (原本0.1)
    
    num_leaves=15,        # 减少叶子，防止过拟合 (原本31)
    max_depth=4,          # 限制深度，只看主要特征 (原本无限制)
    
    min_child_samples=30, # 每个叶子至少要包含30个样本，避免针对个例
    reg_alpha=0.1,        # L1 正则化 (稍微惩罚一下复杂的权重)
    reg_lambda=0.1        # L2 正则化
),
    
    # --- 强力有监督基线 ---
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    
    # --- 简单线性基线 ---
    'Logistic Reg.': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    
    # --- 无监督基线 (海报原本的对比) ---
    # 孤立森林是无监督的，不能用 class_weight，也没有 fit(X, y) 只有 fit(X)
    'Isolation Forest': 'Unsupervised_IF' 
}

# ==========================================
# 4. 训练与评估
# ==========================================
results = {'Model': [], 'AUC': [], 'F1-Score': []}

print("-" * 50)
for name, model in models.items():
    print(f"正在跑模型: {name}...")
    
    if name == 'Isolation Forest':
        # 孤立森林特殊处理 (无监督)
        clf = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        clf.fit(X_train) # 不给 y
        
        # 预测：-1是异常，1是正常。我们需要转换成 0(正常) 和 1(异常)
        y_pred_raw = clf.predict(X_test)
        y_pred = np.where(y_pred_raw == -1, 1, 0)
        # 孤立森林没有标准的 predict_proba，我们用 decision_function 近似
        y_score = -clf.decision_function(X_test) # 越小越异常，所以取负
        
    else:
        # 有监督模型标准流程
        model.fit(X_train, y_train)
        
        # 获取属于类别 1 (中风) 的概率
        y_score = model.predict_proba(X_test)[:, 1]
        
        # 🟢【核心修改】不要直接用 predict()，那是基于 0.5 阈值的
        # 对于不平衡数据，我们把阈值降到 0.2 或 0.15 (根据你的实际情况微调)
        # 意思就是：只要中风概率超过 20%，就判定为中风
        threshold = 0.15  # 你可以试 0.15, 0.2, 0.25
        y_pred = (y_score > threshold).astype(int)

    # 计算指标 (代码不变)
    auc = roc_auc_score(y_test, y_score)
    f1 = f1_score(y_test, y_pred)
    # 计算指标
    auc = roc_auc_score(y_test, y_score)
    f1 = f1_score(y_test, y_pred)
    
    results['Model'].append(name)
    results['AUC'].append(auc)
    results['F1-Score'].append(f1)
    
    print(f"   >> AUC: {auc:.4f} | F1: {f1:.4f}")

# ==========================================
# 5. 绘制海报级对比图 (Bar Chart)
# ==========================================
df_res = pd.DataFrame(results)

# 设置柱状图位置
x = np.arange(len(df_res['Model']))
width = 0.35  # 柱子宽度

fig, ax = plt.subplots()

# 画两组柱子
rects1 = ax.bar(x - width/2, df_res['AUC'], width, label='AUC', color='#2ca02c') # 绿色
rects2 = ax.bar(x + width/2, df_res['F1-Score'], width, label='F1-Score', color='#1f77b4') # 蓝色

# 设置标签和标题
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Comprehensive Model Comparison', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df_res['Model'], fontweight='bold')
ax.set_ylim(0, 1.1) # Y轴稍微留点空
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True) # 图例加个框更清楚

# 给柱子上方标数值 (让评委一眼看到数据)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# 加一条水平线标出 Baseline (比如 0.5)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.text(3.6, 0.51, 'Random Guess', fontsize=12, color='gray')

plt.tight_layout()
plt.savefig("poster_model_comparison_v2.png", dpi=300)
plt.show()

print("-" * 50)
print("✅ 新的对比图已保存为 poster_model_comparison_v2.png")
print("快去看看 LightGBM 是不是遥遥领先！")