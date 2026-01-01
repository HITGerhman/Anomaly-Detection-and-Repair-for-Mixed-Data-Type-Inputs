import streamlit as st
import pandas as pd
import joblib
import os
import shap
import numpy as np

# 导入配置和核心模块
from config import PATHS, FILES
from src.repair_module import AnomalyRepairer
from src.utils import process_and_train, save_system_state, load_system_state

# ==========================================
# 1. 页面配置与状态初始化
# ==========================================
st.set_page_config(page_title="Mixed-Type Anomaly Detection System", layout="wide")

# --- 【关键修改】初始化 Session State (记忆模块) ---
# 如果系统第一次启动，先在内存里建几个"空抽屉"来放数据
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None  # 存放上传的数据
if 'train_metrics' not in st.session_state:
    st.session_state.train_metrics = None # 存放训练分数
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False   # 记录是否训练过

# Sidebar 导航
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["1. Data & Model Training", "2. Detection & Repair"])

# =========================================================
# 页面 1: 数据上传与训练
# =========================================================
if page == "1. Data & Model Training":
    st.title("🛠️ System Setup: Data Import & Training")
    st.markdown("Upload your mixed-type dataset (CSV) to build the anomaly detection model.")
    
    # 1. 文件上传
    # 注意：切换页面后 file_uploader 控件本身会重置，这是 Streamlit 的特性
    # 但我们把读取后的数据存到了 session_state 里，所以数据不会丢
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    # 如果用户刚上传了新文件
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_df = df  # 【存入记忆】
            # 如果上传了新文件，重置训练状态
            st.session_state.is_trained = False 
            st.session_state.train_metrics = None
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # 2. 检查记忆中是否有数据
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        st.success(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head())
        
        # 选择目标列
        target_col = st.selectbox("Select the Target Column (Label)", df.columns, index=len(df.columns)-1)
        st.info(f"The system will learn to detect anomalies based on '{target_col}'. (0=Normal, 1=Anomaly)")
        
        # 3. 训练按钮
        if st.button("🚀 Start Training Model"):
            with st.spinner('Training LightGBM model and preparing repair database...'):
                # 调用 utils
                model, X_test, normal_data, metrics, feats = process_and_train(df, target_col)
                
                # 保存到硬盘（使用配置中的路径）
                save_system_state(model, X_test, normal_data, feats)
                
                # 【存入记忆】
                st.session_state.train_metrics = metrics
                st.session_state.is_trained = True
                
            st.success("✅ Training Complete!")
            st.balloons()

    # 4. 显示训练结果 (即使刷新页面，只要 session_state 里有，就显示)
    if st.session_state.is_trained and st.session_state.train_metrics is not None:
        st.markdown("---")
        st.subheader("📊 Model Performance")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("F1-Score", f"{st.session_state.train_metrics['f1']:.4f}")
        col_m2.metric("AUC-ROC", f"{st.session_state.train_metrics['auc']:.4f}")
        
        st.markdown("👉 **Now go to '2. Detection & Repair' page to test the system.**")

# =========================================================
# 页面 2: 检测与修复
# =========================================================
elif page == "2. Detection & Repair":
    st.title("🔍 Interactive Detection & Repair")
    
    # 检查硬盘上有没有模型文件 (这是为了防止用户直接跳到这一页)
    if not os.path.exists(FILES["model"]):
        st.warning("⚠️ No model found. Please go to 'Data & Model Training' page first.")
        st.stop()
        
    # 加载模型 (使用 cache_resource 避免重复加载)
    @st.cache_resource
    def load_model_resources():
        return load_system_state()

    model, X_test, normal_data = load_model_resources()
    
    # 初始化修复器
    if 'repairer' not in st.session_state:
        st.session_state.repairer = AnomalyRepairer(normal_data)
        
    st.sidebar.markdown("---")
    st.sidebar.header("Test Console")
    
    # 防止滑块报错 (如果新数据比旧数据小)
    max_len = len(X_test) - 1
    if max_len < 0: max_len = 0
    
    sample_id = st.sidebar.number_input(
        "Enter Test Sample ID", 
        min_value=0, 
        max_value=max_len, 
        value=0, 
        step=1,
        help=f"Valid range: 0 to {max_len}" # 鼠标悬停会提示范围
    )
    
    # --- 检测逻辑 ---
    try:
        sample_data = X_test.iloc[[sample_id]]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Target Profile")
            st.dataframe(sample_data.T, height=400)
            
        with c2:
            st.subheader("Analysis Result")
            # 自动运行或者手动运行，这里用按钮更清晰
            if st.button("Run Diagnosis", key="run_diag"):
                pred = model.predict(sample_data)[0]
                prob = model.predict_proba(sample_data)[0][1]
                
                if pred == 1: # 异常
                    st.error(f"🚨 ANOMALY DETECTED (Risk Score: {prob:.4f})")
                    
                    # SHAP
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(sample_data)
                    vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                    
                    top_indices = np.argsort(vals)[::-1][:3]
                    feat_names = sample_data.columns
                    
                    st.markdown("### 🛠️ Smart Repair Suggestions")
                    
                    for idx in top_indices:
                        if vals[idx] > 0:
                            fname = feat_names[idx]
                            report, _ = st.session_state.repairer.generate_repair_suggestion(sample_data, fname)
                            
                            with st.expander(f"🔴 Issue: {fname} (Impact: +{vals[idx]:.2f})", expanded=True):
                                st.write(f"**Current:** {sample_data[fname].values[0]}")
                                st.success(f"**Suggested:** {report['Suggested Value']}")
                                st.caption(f"Reasoning: {report['Repair Logic']}")
                                
                else:
                    st.success(f"✅ Normal Profile (Risk Score: {prob:.4f})")
    except Exception as e:
        st.error(f"Error analyzing sample: {e}")