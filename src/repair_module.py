import pandas as pd
import numpy as np
import gower

class AnomalyRepairer:
    def __init__(self, normal_data, feature_weights=None):
        """
        初始化修复器
        :param normal_data: 正常的样本库 (DataFrame)
        :param feature_weights: 特征权重数组
        """
        # 1. 自动识别哪些列是"分类" (Category)
        # 我们检查每一列的类型，如果是 'category' 或者 'object'，就标记为 True
        self.cat_features = [
            (pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype))
            for dtype in normal_data.dtypes
        ]
        
        # 2. 制作一个"干净版"的数据矩阵给 Gower 用
        # Gower 不喜欢 pandas 的 Category 类型，所以我们把它们转成纯数字编码 (.cat.codes)
        self.normal_data_matrix = normal_data.copy()
        for col in self.normal_data_matrix.columns:
            if pd.api.types.is_categorical_dtype(self.normal_data_matrix[col]):
                self.normal_data_matrix[col] = self.normal_data_matrix[col].cat.codes
        
        # 3. 保存原始数据 (为了最后给建议时能取到真实的值)
        self.normal_data = normal_data
        self.feature_weights = feature_weights

    def generate_repair_suggestion(self, anomaly_sample, feature_to_fix, k=5):
        """
        基于加权 KNN + Gower 距离生成修复建议
        """
        # 1. 同样处理异常样本：把 Category 类型转成纯数字编码
        anomaly_sample_matrix = anomaly_sample.copy()
        for col in anomaly_sample_matrix.columns:
            if pd.api.types.is_categorical_dtype(anomaly_sample_matrix[col]):
                anomaly_sample_matrix[col] = anomaly_sample_matrix[col].cat.codes
        
        # 2. 计算距离 (使用处理过的矩阵)
        # 关键点：我们传入了 cat_features，告诉 Gower 哪些列是分类特征
        # 这样 Gower 就会用 Dice 距离(0或1)而不是曼哈顿距离
        distances = gower.gower_matrix(
            anomaly_sample_matrix, 
            self.normal_data_matrix, 
            weight=self.feature_weights,
            cat_features=self.cat_features # 【核心修复】显式指定分类列
        )[0]
        
        # 3. 找到最近的 K 个邻居的索引
        nearest_indices = distances.argsort()[:k]
        
        # 4. 从【原始数据】(self.normal_data) 中取出邻居
        # 注意：这里必须用原始数据，因为我们需要拿到原始的分类标签(比如 'Private')，而不是数字编码
        neighbors = self.normal_data.iloc[nearest_indices]
        
        # 5. 计算建议值
        target_col_values = neighbors[feature_to_fix]
        
        if pd.api.types.is_numeric_dtype(target_col_values):
            suggested_value = target_col_values.median()
            if suggested_value.is_integer():
                suggested_value = int(suggested_value)
            else:
                suggested_value = round(suggested_value, 2)
        else:
            # 类别型取众数
            suggested_value = target_col_values.mode()[0]
            
        repair_logic = f"Found {k} nearest healthy neighbors (weighted by SHAP importance)."
        
        return {
            "Suggested Value": suggested_value,
            "Repair Logic": repair_logic,
            "Neighbors": neighbors
        }, neighbors