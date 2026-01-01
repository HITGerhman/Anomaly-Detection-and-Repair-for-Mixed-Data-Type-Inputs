import pandas as pd
import lightgbm as lgb
import shap
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score

def process_and_train(df, target_col):
    """
    接收用户上传的DataFrame，进行清洗、编码、训练，并返回模型和评估指标
    """
    # 1. 数据预处理
    # 自动识别类型
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 简单的标签编码 (针对文字列) - 为了通用性，这里用 LabelEncoder 简化处理
    encoders = {}
    X_encoded = X.copy()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        le = LabelEncoder()
        # 强制转字符串处理缺失值
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        # 记得把列类型转为 category 给 LightGBM 用 (虽然上面encoded了，但LGBM喜欢category类型)
        X_encoded[col] = X_encoded[col].astype('category')

    # 2. 划分训练集
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # 3. 训练 LightGBM
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # 4. 评估
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        "f1": f1_score(y_test, y_pred, average='weighted'), # weighted 兼容多分类或不平衡
        "auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else 0.0 # 只有二分类才算AUC
    }
    
    # 5. 保存正常样本库 (给 KNN 修复用)
    # 假设标签 0 是正常
    normal_data = X_train[y_train == 0].copy()
    
    return model, X_test, normal_data, metrics, X.columns.tolist()

def save_system_state(model, X_test, normal_data, feature_names, save_dir=r"D:\code\pythoncode"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    joblib.dump(model, os.path.join(save_dir, "model_lgb.pkl"))
    joblib.dump(X_test, os.path.join(save_dir, "test_data.pkl"))
    joblib.dump(normal_data, os.path.join(save_dir, "normal_data.pkl"))
    
    # 保存列名配置
    config = {"feature_names": feature_names}
    joblib.dump(config, os.path.join(save_dir, "config.pkl"))