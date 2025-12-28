import streamlit as st
import pandas as pd
import joblib
import os
import shap
import numpy as np
from PIL import Image
from repair_module import AnomalyRepairer  # 导入我们刚才写的修复模块

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="Anomaly Detection System", layout="wide")
st.title("🔍 Mixed-Type Data Anomaly Detection System")
st.markdown("**Core Framework:** LightGBM + SHAP + KNN-Repair | **Status:** v2.0 Integrated")
st.markdown("---")

# ==========================================
# 2. 加载资源
# ==========================================
base_dir = r"D:\code\pythoncode"

@st.cache_resource
def load_resources():
    model = joblib.load(os.path.join(base_dir, "model_lgb.pkl"))
    data = joblib.load(os.path.join(base_dir, "test_data.pkl"))
    normal_data = joblib.load(os.path.join(base_dir, "normal_data.pkl")) # 加载正常样本库
    return model, data, normal_data

try:
    model, X_test, normal_data = load_resources()
    # 初始化修复器 (只做一次)
    if 'repairer' not in st.session_state:
        st.session_state.repairer = AnomalyRepairer(normal_data)
    st.sidebar.success(f"✅ System Online. Reference DB: {len(normal_data)} samples")
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# ==========================================
# 3. 侧边栏与样本选择
# ==========================================
st.sidebar.header("Control Panel")
# 为了方便演示，我把几个必定异常的 ID 列在这里，省得你找
st.sidebar.info("Hint: Try Sample ID 4, 11, or 82 to see anomalies.")
sample_id = st.sidebar.number_input("Select Sample ID", min_value=0, max_value=len(X_test)-1, value=4)
sample_data = X_test.iloc[[sample_id]]

# ==========================================
# 4. 主界面
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Incoming Data")
    st.dataframe(sample_data.T, height=300)

with col2:
    st.subheader("2. Detection & Diagnosis")
    
    if st.button("🚀 Run Analysis"):
        # --- A. 检测 (Detection) ---
        prediction = model.predict(sample_data)[0]
        prob = model.predict_proba(sample_data)[0][1]
        
        # 模拟计算进度
        import time
        my_bar = st.progress(0)
        for p in range(50):
            time.sleep(0.01)
            my_bar.progress(p + 1)
            
        if prediction == 0:
            my_bar.progress(100)
            st.success(f"✅ Normal Sample (Anomaly Score: {prob:.4f})")
            st.info("No repair needed.")
        else:
            # 异常情况！
            my_bar.progress(100)
            st.error(f"🚨 ANOMALY DETECTED (Score: {prob:.4f})")
            
            # --- B. 诊断 (SHAP Explanation) ---
            st.write("---")
            st.subheader("3. Root Cause & Repair Suggestions")
            st.write("Analyzing contributing factors...")
            
            # 1. 现场计算 SHAP 值 (找出是谁导致了异常)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_data)
            
            # 兼容处理：LightGBM Binary分类有时返回list，有时返回array
            if isinstance(shap_values, list):
                # 如果是列表，取索引1 (Positive class/Anomaly)
                vals = shap_values[1][0]
            else:
                vals = shap_values[0]
            
            # 2. 找出影响最大的 3 个特征 (SHAP值越大，说明越推高异常分)
            feature_names = sample_data.columns
            # argsort 从小到大排，[::-1] 反转变成从大到小
            top_indices = np.argsort(vals)[::-1]
            
            # 3. 逐个生成修复建议
            repair_cols = st.columns(3)
            count = 0
            
            for idx in top_indices:
                if count >= 3: break # 只显示前3个主要原因
                
                # 只关心正向贡献的特征 (真正导致异常的)
                if vals[idx] > 0:
                    feature_name = feature_names[idx]
                    
                    # --- C. 修复 (Repair) ---
                    # 调用我们写的 repair_module
                    report, _ = st.session_state.repairer.generate_repair_suggestion(sample_data, feature_name)
                    
                    with repair_cols[count]:
                        st.markdown(f"**🔴 Issue: {feature_name}**")
                        st.caption(f"Contribution: +{vals[idx]:.2f}")
                        
                        st.markdown("---")
                        st.markdown("**🛠️ Suggestion:**")
                        # 重点高亮建议值
                        st.success(f"{report['Suggested Value']}")
                        st.caption(f"Ref: 5 similar normal profiles")
                    
                    count += 1