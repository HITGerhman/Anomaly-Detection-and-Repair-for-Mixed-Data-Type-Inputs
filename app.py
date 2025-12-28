import streamlit as st
import pandas as pd
import joblib
import os
import shap
import numpy as np
from PIL import Image
from repair_module import AnomalyRepairer

# ==========================================
# 1. 基础页面设置
# ==========================================
st.set_page_config(page_title="Intelligent Anomaly Detection", layout="wide")

# ==========================================
# 2. 加载资源与配置
# ==========================================
base_dir = r"D:\code\pythoncode"

@st.cache_resource
def load_resources():
    model = joblib.load(os.path.join(base_dir, "model_lgb.pkl"))
    data = joblib.load(os.path.join(base_dir, "test_data.pkl"))
    normal_data = joblib.load(os.path.join(base_dir, "normal_data.pkl"))
    # 读取配置文件
    config = joblib.load(os.path.join(base_dir, "config.pkl"))
    return model, data, normal_data, config

try:
    model, X_test, normal_data, config = load_resources()
    
    # 初始化修复器
    if 'repairer' not in st.session_state:
        st.session_state.repairer = AnomalyRepairer(normal_data)
        
    dataset_name = config.get("dataset_name", "unknown")
    
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# ==========================================
# 3. 动态标题与侧边栏 (自适应切换)
# ==========================================

# 根据数据集名称，显示不同的标题
if dataset_name == "stroke":
    st.title("🏥 AI Stroke Risk Prediction System")
    st.markdown("**Dataset:** Real-world Healthcare Data | **Model:** LightGBM + SHAP")
    st.sidebar.success("Mode: Medical / Stroke Analysis")
    # 医疗数据的异常提示
    anomaly_msg = "⚠️ HIGH STROKE RISK DETECTED"
    normal_msg = "✅ Low Risk / Healthy Profile"
    
elif dataset_name == "adult":
    st.title("💰 Census Income Anomaly Detection")
    st.markdown("**Dataset:** Adult Census Data | **Model:** LightGBM + SHAP")
    st.sidebar.success("Mode: Financial / Census Analysis")
    # 收入数据的异常提示
    anomaly_msg = "🚨 ANOMALY DETECTED (High Income)"
    normal_msg = "✅ Normal Profile"
    
else:
    st.title("🔍 Anomaly Detection System")
    st.sidebar.warning("Unknown Dataset Mode")
    anomaly_msg = "🚨 ANOMALY DETECTED"
    normal_msg = "✅ Normal"

st.markdown("---")

# ==========================================
# 4. 控制面板
# ==========================================
st.sidebar.header("Control Panel")
# 动态获取样本总数
max_idx = len(X_test) - 1
st.sidebar.info(f"Test Set Size: {len(X_test)} samples")

sample_id = st.sidebar.number_input(f"Select Sample ID (0-{max_idx})", min_value=0, max_value=max_idx, value=0)
sample_data = X_test.iloc[[sample_id]]

# ==========================================
# 5. 主界面逻辑 (通用)
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Patient / User Profile")
    st.dataframe(sample_data.T, height=400)

with col2:
    st.subheader("2. AI Diagnosis")
    
    if st.button("🚀 Run Analysis"):
        # A. 预测
        prediction = model.predict(sample_data)[0]
        prob = model.predict_proba(sample_data)[0][1]
        
        # 进度条
        import time
        my_bar = st.progress(0)
        for p in range(50):
            time.sleep(0.01)
            my_bar.progress(p + 1)
        my_bar.progress(100)
            
        # B. 结果显示 (使用上面的动态文案)
        if prediction == 0:
            st.success(f"{normal_msg} (Score: {prob:.4f})")
        else:
            st.error(f"{anomaly_msg} (Score: {prob:.4f})")
            
            # C. 解释与修复
            st.write("---")
            st.subheader("3. Risk Factors & Suggestions")
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_data)
            
            if isinstance(shap_values, list):
                vals = shap_values[1][0]
            else:
                vals = shap_values[0]
            
            feature_names = sample_data.columns
            top_indices = np.argsort(vals)[::-1]
            
            # 显示修复建议卡片
            repair_cols = st.columns(3)
            count = 0
            
            for idx in top_indices:
                if count >= 3: break
                if vals[idx] > 0: # 只关注推高风险的因素
                    feat_name = feature_names[idx]
                    
                    # 调用修复模块
                    report, _ = st.session_state.repairer.generate_repair_suggestion(sample_data, feat_name)
                    
                    with repair_cols[count]:
                        st.markdown(f"**🔴 Factor: {feat_name}**")
                        st.caption(f"Impact: +{vals[idx]:.2f}")
                        st.markdown("---")
                        st.markdown("**🩺 Advice:**")
                        st.success(f"{report['Suggested Value']}")
                        st.caption("Based on similar healthy profiles")
                    
                    count += 1