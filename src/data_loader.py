"""
数据加载模块
支持加载和预处理各种数据集
"""
import pandas as pd
import numpy as np
import os
import sys

# 导入配置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FILES


def load_stroke_data(filepath=None):
    """
    加载并清洗中风预测数据集
    
    Args:
        filepath: 数据文件路径（默认使用配置文件中的路径）
    
    Returns:
        tuple: (X, y) 特征和标签
    """
    if filepath is None:
        filepath = FILES["stroke_data"]
    
    print(f"正在加载真实数据集: {filepath} ...")
    
    # 1. 读取 CSV
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件 {filepath}，请确保你已经下载并放到了项目目录下！")

    # 2. 数据清洗
    # drop 'id': ID列对预测没用，还会干扰模型
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 处理缺失值: 'bmi' 列里面有一些 NaN
    # 简单策略：用平均值填充
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
    
    # 处理 'N/A' 或 'Unknown' (在吸烟状态里有 Unknown，我们暂时保留它作为一个类别)

    # 3. 分离特征 (X) 和 标签 (y)
    # stroke 列是目标：1表示中风(异常)，0表示健康
    y = df['stroke']
    X = df.drop(columns=['stroke'])

    # 4. 类型转换
    # LightGBM 需要把文字列强制转为 'category' 类型
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')

    print(f"数据加载完成！样本数: {len(X)}, 特征数: {X.shape[1]}")
    print(f"异常样本比例: {y.mean():.2%}")
    
    return X, y

if __name__ == "__main__":
    # 测试一下
    X, y = load_stroke_data()
    print(X.head())