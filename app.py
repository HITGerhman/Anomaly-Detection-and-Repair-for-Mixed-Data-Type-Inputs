import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# 1. 页面基础设置
# ==========================================
st.set_page_config(page_title="Anomaly Detection System", layout="wide")

# 标题和介绍
st.title("🔍 Mixed-Type Data Anomaly Detection System")
st.markdown("**Core Framework:** LightGBM + SHAP | **Status:** Prototype v1.0")
st.markdown("---")

# ==========================================
# 2. 加载资源 (模型、数据、图片)
# ==========================================
# 这里的路径对应你刚才保存的位置，如果在同一文件夹下不用改
base_dir = r"D:\code\pythoncode"

@st.cache_resource  # 缓存机制，让网页加载更快
def load_resources():
    model = joblib.load(os.path.join(base_dir, "model_lgb.pkl"))
    data = joblib.load(os.path.join(base_dir, "test_data.pkl"))
    return model, data

try:
    model, X_test = load_resources()
    st.sidebar.success("✅ System Online: Model Loaded")
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# ==========================================
# 3. 侧边栏控制区
# ==========================================
st.sidebar.header("Control Panel")
st.sidebar.info("Select a sample from the test dataset to simulate real-time detection.")

# 让用户选择一个样本进行检测
sample_id = st.sidebar.slider("Select Sample ID", 0, 100, 0)
sample_data = X_test.iloc[[sample_id]]

# ==========================================
# 4. 主界面：展示数据与检测结果
# ==========================================

# 分两列展示
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Incoming Data Stream")
    st.write("Current sample features:")
    # 转置显示，看起来更像“个人档案”
    st.dataframe(sample_data.T, height=400)

with col2:
    st.subheader("2. Detection Result")
    
    if st.button("🚀 Run Anomaly Detection"):
        # 预测
        prediction = model.predict(sample_data)[0]
        prob = model.predict_proba(sample_data)[0][1]
        
        # 模拟进度条
        import time
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
            
        # 显示结果
        if prediction == 1: # 假设 1 是高收入/异常
            st.error(f"🚨 ALERT: Anomaly Detected! (Score: {prob:.4f})")
        else:
            st.success(f"✅ Normal: Data is within safe range. (Score: {prob:.4f})")
            
        st.subheader("3. Model Explanation (Global)")
        st.write("Top contributing features based on SHAP values:")
        
        # 显示我们之前生成的静态图片
        img_path = os.path.join(base_dir, "result_shap_importance.png")
        if os.path.exists(img_path):
            image = Image.open(img_path)
            st.image(image, caption="Feature Importance (Global Interpretation)", use_container_width=True)
        else:
            st.warning("Analysis chart not found.")