import pandas as pd
import numpy as np
import gower  # 专门计算混合数据距离的库
from scipy.stats import mode

class AnomalyRepairer:
    def __init__(self, normal_data):
        """
        初始化修复器
        :param normal_data: 这里的输入必须是【只包含正常样本】的 DataFrame
        """
        self.normal_data = normal_data.reset_index(drop=True)
        print(f"🔧 修复模块已初始化 | 参考库大小: {len(self.normal_data)} 条正常数据")

    def find_neighbors(self, anomaly_sample, k=5):
        """
        计算 Gower 距离并找到最近的 k 个邻居
        """
        # 1. 计算 Gower 距离矩阵
        # gower.gower_matrix 会自动识别数字列和文字列
        # 返回的是一个矩阵，我们只需要第一行（因为只有一个异常样本）
        distances = gower.gower_matrix(anomaly_sample, self.normal_data)[0]
        
        # 2. 找到距离最小的 k 个索引 (argsort 从小到大排序)
        # 注意：Gower 距离 0 表示完全一样，1 表示完全不同
        nearest_indices = np.argsort(distances)[:k]
        
        # 3. 提取这 k 个邻居的数据
        neighbors = self.normal_data.iloc[nearest_indices]
        return neighbors, distances[nearest_indices]

    def generate_repair_suggestion(self, anomaly_sample, target_feature, k=5):
        """
        针对某个特定特征（target_feature）生成修复建议
        """
        # 1. 找邻居
        neighbors, dists = self.find_neighbors(anomaly_sample, k)
        
        # 2. 获取邻居在该特征上的值
        neighbor_values = neighbors[target_feature]
        
        # 3. 判断特征类型（是数字还是文字？）
        # pandas 的 api: api.types.is_numeric_dtype
        is_numeric = pd.api.types.is_numeric_dtype(neighbor_values)
        
        current_value = anomaly_sample[target_feature].values[0]
        
        if is_numeric:
            # 如果是数字，算平均值 (Mean)
            suggested_value = neighbor_values.mean()
            # 格式化一下，保留2位小数
            suggestion_text = f"{suggested_value:.2f} (Mean of neighbors)"
            repair_value = suggested_value
        else:
            # 如果是文字，算众数 (Mode) - 也就是出现次数最多的
            # mode result 返回 (array([值]), array([次数]))
            mode_res = mode(neighbor_values, keepdims=True)
            suggested_value = mode_res.mode[0]
            count = mode_res.count[0]
            suggestion_text = f"'{suggested_value}' (Mode, appeared {count}/{k} times)"
            repair_value = suggested_value

        # 4. 生成最终报告
        report = {
            "Feature": target_feature,
            "Current Value": current_value,
            "Suggested Value": suggestion_text,
            "Repair Logic": f"Based on {k} most similar normal samples (Avg Gower Dist: {dists.mean():.4f})",
            "Raw_Repair_Value": repair_value # 用于程序后续自动替换
        }
        
        return report, neighbors

# ==========================================
# 下面是测试代码 (Test Block)
# ==========================================
if __name__ == "__main__":
    import shap
    from sklearn.model_selection import train_test_split

    print("--- 开始测试修复模块 ---")
    
    # 1. 准备数据 (还是用 Adult 数据集)
    X, y = shap.datasets.adult()
    # 假设标签为 False (0) 是正常人，True (1) 是异常/高收入
    # 我们只用"正常人"作为参考库
    normal_data_pool = X[y == False].sample(1000, random_state=42) # 取1000个做演示，太大数据算得慢
    
    # 找一个"异常"样本 (假设 y==True 的是异常)
    anomaly_sample = X[y == True].iloc[[0]] 
    
    # 2. 实例化修复器
    repairer = AnomalyRepairer(normal_data_pool)
    
    # 3. 假设 SHAP 告诉我们要修复 "Age" 和 "Relationship"
    target_features = ["Age", "Relationship"]
    
    print(f"\n当前异常样本:\n{anomaly_sample.iloc[0][target_features].to_dict()}")
    print("-" * 50)
    
    for feature in target_features:
        print(f"正在计算 {feature} 的修复建议...")
        report, neighbors = repairer.generate_repair_suggestion(anomaly_sample, feature, k=5)
        
        print(f"✅ 针对 [{feature}] 的修复建议:")
        print(f"   - 原值: {report['Current Value']}")
        print(f"   - 建议修改为: {report['Suggested Value']}")
        print(f"   - 依据: {report['Repair Logic']}")
        print("-" * 50)