"""
查找异常样本脚本
扫描测试集，找出所有被模型判定为异常的样本
"""
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_system_state

print("📂 正在加载测试数据和模型...")

# 1. 加载测试集和模型
try:
    model, X_test, normal_data = load_system_state()
except Exception as e:
    print(f"❌ 错误: 找不到文件。请确保你已经运行过 'Data & Model Training' 页面。报错: {e}")
    exit()

# 2. 全量预测
print(f"🔍 正在扫描 {len(X_test)} 个样本...")
y_pred = model.predict(X_test)

# 3. 找出所有被判为"异常 (1)"的索引
anomaly_indices = np.where(y_pred == 1)[0]

# 4. 打印结果
print("\n" + "="*40)
print(f"🚨 成功发现 {len(anomaly_indices)} 个异常样本！")
print("="*40)
print("请在 App 的 'Select Test Sample ID' 中输入以下任意一个数字：\n")

# 每行打印 10 个
for i in range(0, len(anomaly_indices), 10):
    print(anomaly_indices[i:i+10])

print("\n" + "="*40)
print("💡 演示建议：")
print("选一个靠前的 ID（比如列表里的第一个），在 App 里选中它，确认它是红色的。")
print("然后看看它的特征（比如是不是年龄大、血糖高），想好怎么解释。")

# 运行方式: python scripts/find_anomalies.py
