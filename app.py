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
        
        # =========================================================
        # 📊 数据统计面板
        # =========================================================
        with st.expander("📊 Data Statistics & Quality Report", expanded=False):
            # --- 基础信息 ---
            st.markdown("#### 📋 Basic Information")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{df.shape[0]:,}")
            col2.metric("Total Columns", df.shape[1])
            col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            col4.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            st.markdown("---")
            
            # --- 数据类型分布 ---
            st.markdown("#### 🏷️ Data Types Distribution")
            dtype_counts = df.dtypes.astype(str).value_counts()
            col_type1, col_type2 = st.columns(2)
            
            with col_type1:
                # 数值型列
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                st.markdown(f"**Numeric Columns** ({len(numeric_cols)})")
                if numeric_cols:
                    st.write(", ".join(numeric_cols))
                else:
                    st.write("None")
            
            with col_type2:
                # 分类型列
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                st.markdown(f"**Categorical Columns** ({len(cat_cols)})")
                if cat_cols:
                    st.write(", ".join(cat_cols))
                else:
                    st.write("None")
            
            st.markdown("---")
            
            # --- 缺失值分析 ---
            st.markdown("#### ❓ Missing Values Analysis")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': (missing_data.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No missing values found!")
            
            st.markdown("---")
            
            # --- 数值特征统计 ---
            if numeric_cols:
                st.markdown("#### 📈 Numeric Features Statistics")
                st.dataframe(df[numeric_cols].describe().T.round(2), use_container_width=True)
            
            # --- 分类特征分布 ---
            if cat_cols:
                st.markdown("#### 📊 Categorical Features Distribution")
                selected_cat = st.selectbox("Select a categorical column to view distribution:", cat_cols)
                if selected_cat:
                    value_counts = df[selected_cat].value_counts()
                    dist_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(df) * 100).round(2)
                    })
                    st.dataframe(dist_df, use_container_width=True, hide_index=True)
        
        # 选择目标列
        target_col = st.selectbox("Select the Target Column (Label)", df.columns, index=len(df.columns)-1)
        
        # --- 目标列分布预览 ---
        if target_col:
            target_counts = df[target_col].value_counts()
            col_t1, col_t2, col_t3 = st.columns(3)
            
            total = len(df)
            normal_count = target_counts.get(0, 0)
            anomaly_count = target_counts.get(1, 0)
            
            col_t1.metric("Normal (0)", f"{normal_count:,}", f"{normal_count/total*100:.1f}%")
            col_t2.metric("Anomaly (1)", f"{anomaly_count:,}", f"{anomaly_count/total*100:.1f}%")
            col_t3.metric("Imbalance Ratio", f"1:{normal_count//max(anomaly_count,1)}" if anomaly_count > 0 else "N/A")
            
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
    
    # ⚡ SHAP Explainer 缓存（性能优化）
    # TreeExplainer 创建开销大，缓存后只创建一次
    @st.cache_resource
    def get_shap_explainer(_model):
        """缓存 SHAP explainer，避免重复创建"""
        return shap.TreeExplainer(_model)
    
    explainer = get_shap_explainer(model)
    
    # 初始化修复器
    if 'repairer' not in st.session_state:
        st.session_state.repairer = AnomalyRepairer(normal_data)
        
    # 防止索引报错
    max_len = len(X_test) - 1
    if max_len < 0: max_len = 0
    
    # =========================================================
    # 使用 Tabs 区分单条检测和批量检测
    # =========================================================
    tab1, tab2 = st.tabs(["🔬 Single Detection", "📊 Batch Detection & Export"])
    
    # ---------------------------------------------------------
    # Tab 1: 单条检测
    # ---------------------------------------------------------
    with tab1:
        st.sidebar.markdown("---")
        st.sidebar.header("🔬 Single Detection")
        
        sample_id = st.sidebar.number_input(
            "Enter Test Sample ID", 
            min_value=0, 
            max_value=max_len, 
            value=0, 
            step=1,
            help=f"Valid range: 0 to {max_len}"
        )
        
        try:
            sample_data = X_test.iloc[[sample_id]]
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Target Profile")
                st.dataframe(sample_data.T, height=400)
                
            with c2:
                st.subheader("Analysis Result")
                if st.button("Run Diagnosis", key="run_diag"):
                    pred = model.predict(sample_data)[0]
                    prob = model.predict_proba(sample_data)[0][1]
                    
                    if pred == 1:
                        st.error(f"🚨 ANOMALY DETECTED (Risk Score: {prob:.4f})")
                        
                        # SHAP 解释（使用缓存的 explainer）
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
    
    # ---------------------------------------------------------
    # Tab 2: 批量检测 + 导出
    # ---------------------------------------------------------
    with tab2:
        st.markdown("### 📊 Batch Anomaly Detection")
        st.markdown("Scan multiple samples at once and export results to CSV.")
        
        # 选择检测范围
        col_range1, col_range2 = st.columns(2)
        with col_range1:
            detection_mode = st.radio(
                "Detection Scope",
                ["All Test Samples", "Custom Range"],
                horizontal=True
            )
        
        if detection_mode == "Custom Range":
            with col_range2:
                range_start = st.number_input("Start Index", min_value=0, max_value=max_len, value=0)
                range_end = st.number_input("End Index", min_value=0, max_value=max_len, value=min(100, max_len))
        else:
            range_start, range_end = 0, max_len
        
        # 批量检测按钮
        if st.button("🚀 Run Batch Detection", key="batch_detect", type="primary"):
            with st.spinner(f"Scanning samples {range_start} to {range_end}..."):
                # 获取指定范围的数据
                batch_data = X_test.iloc[range_start:range_end+1]
                
                # 批量预测
                predictions = model.predict(batch_data)
                probabilities = model.predict_proba(batch_data)[:, 1]
                
                # 构建结果 DataFrame
                results_df = batch_data.copy()
                results_df.insert(0, 'Sample_ID', range(range_start, range_end+1))
                results_df['Prediction'] = predictions
                results_df['Risk_Score'] = probabilities.round(4)
                results_df['Status'] = np.where(predictions == 1, '🚨 Anomaly', '✅ Normal')
                
                # 保存到 session_state
                st.session_state.batch_results = results_df
                st.session_state.batch_stats = {
                    'total': len(results_df),
                    'anomalies': int((predictions == 1).sum()),
                    'normals': int((predictions == 0).sum())
                }
            
            st.success("✅ Batch detection complete!")
        
        # 显示结果
        if 'batch_results' in st.session_state and st.session_state.batch_results is not None:
            stats = st.session_state.batch_stats
            results_df = st.session_state.batch_results
            
            # 统计指标
            st.markdown("---")
            st.markdown("### 📈 Detection Summary")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Total Scanned", f"{stats['total']:,}")
            col_s2.metric("Anomalies Found", f"{stats['anomalies']:,}", 
                         delta=f"{stats['anomalies']/stats['total']*100:.1f}%", delta_color="inverse")
            col_s3.metric("Normal Samples", f"{stats['normals']:,}")
            col_s4.metric("Anomaly Rate", f"{stats['anomalies']/stats['total']*100:.2f}%")
            
            st.markdown("---")
            
            # 筛选选项
            filter_option = st.radio(
                "Filter Results",
                ["All", "Anomalies Only", "Normal Only"],
                horizontal=True
            )
            
            if filter_option == "Anomalies Only":
                display_df = results_df[results_df['Prediction'] == 1]
            elif filter_option == "Normal Only":
                display_df = results_df[results_df['Prediction'] == 0]
            else:
                display_df = results_df
            
            # 显示结果表格
            st.markdown(f"### 📋 Results ({len(display_df)} samples)")
            st.dataframe(
                display_df[['Sample_ID', 'Status', 'Risk_Score'] + list(X_test.columns)],
                use_container_width=True,
                height=400
            )
            
            st.markdown("---")
            
            # 导出功能
            st.markdown("### 📥 Export Results")
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                # 导出全部结果
                csv_all = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download All Results (CSV)",
                    data=csv_all,
                    file_name="batch_detection_all.csv",
                    mime="text/csv",
                    key="download_all"
                )
            
            with col_exp2:
                # 只导出异常
                anomalies_df = results_df[results_df['Prediction'] == 1]
                if len(anomalies_df) > 0:
                    csv_anomalies = anomalies_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="🚨 Download Anomalies Only (CSV)",
                        data=csv_anomalies,
                        file_name="batch_detection_anomalies.csv",
                        mime="text/csv",
                        key="download_anomalies"
                    )
                else:
                    st.info("No anomalies found to export.")