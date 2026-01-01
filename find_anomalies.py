import joblib
import os
import numpy as np
import pandas as pd

# 设置路径 (保持和你 app.py 里的一致)
base_dir = r"D:\code\pythoncode\Anomaly Detection and Repair for Mixed Data Type Inputs"

print("📂 正在加载测试数据和模型...")

# 1. 加载测试集和模型
try:
    X_test = joblib.load(os.path.join(base_dir, "test_data.pkl"))
    model = joblib.load(os.path.join(base_dir, "model_lgb.pkl"))
except Exception as e:
    print(f"❌ 错误: 找不到文件。请确保你已经运行过 'Data & Model Training' 页面。报错: {e}")
    exit()

# 2. 全量预测
print("🔍 正在扫描 1021 个样本...")
y_pred = model.predict(X_test)

# 3. 找出所有被判为"异常 (1)"的索引
# np.where 返回的是下标，刚好对应你 App 滑块的 "Sample ID"
anomaly_indices = np.where(y_pred == 1)[0]

# 4. 打印结果
print("\n" + "="*40)
print(f"🚨 成功发现 {len(anomaly_indices)} 个异常样本！")
print("="*40)
print("请在 App 的 'Select Test Sample ID' 滑块中选择以下任意一个数字：\n")

# 为了方便看，我们每行打印 10 个
for i in range(0, len(anomaly_indices), 10):
    print(anomaly_indices[i:i+10])

print("\n" + "="*40)
print("💡 演示建议：")
print("选一个靠前的 ID（比如列表里的第一个），在 App 里先选中它，确认它是红色的。")
print("然后看看它的特征（比如是不是年龄大、血糖高），想好怎么解释。")
#py -3.9 find_anomalies.py