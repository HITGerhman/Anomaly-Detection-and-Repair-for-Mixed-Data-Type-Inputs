import streamlit as st
import pandas as pd
import joblib
import os
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 导入配置和核心模块
from config import PATHS, FILES
from src.repair_module import AnomalyRepairer
from src.utils import (
    load_system_state,
    predict_with_threshold,
    process_and_train,
    save_system_state,
)

# 设置 matplotlib 样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

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
        metrics = st.session_state.train_metrics
        
        st.markdown("---")
        st.subheader("📊 Model Performance")
        
        # 指标卡片 - 5个核心指标
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        col_m2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        col_m3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        col_m4.metric("F1-Score", f"{metrics['f1']:.4f}")
        col_m5.metric("AUC-ROC", f"{metrics['auc']:.4f}")
        
        st.markdown("---")
        
        # =========================================================
        # 📈 可视化图表区域
        # =========================================================
        st.subheader("📈 Visual Analytics")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🎯 ROC Curve", "📊 Confusion Matrix", "⭐ Feature Importance"])
        
        # ---------------------------------------------------------
        # ROC 曲线
        # ---------------------------------------------------------
        with viz_tab1:
            if "roc_curve" in metrics:
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                
                fpr = metrics["roc_curve"]["fpr"]
                tpr = metrics["roc_curve"]["tpr"]
                auc_score = metrics["auc"]
                
                # 绘制 ROC 曲线
                ax_roc.plot(fpr, tpr, color='#3498db', lw=2.5, 
                           label=f'ROC Curve (AUC = {auc_score:.4f})')
                ax_roc.plot([0, 1], [0, 1], color='#95a5a6', lw=1.5, 
                           linestyle='--', label='Random Classifier')
                
                # 填充 AUC 区域
                ax_roc.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
                
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate', fontsize=12)
                ax_roc.set_ylabel('True Positive Rate', fontsize=12)
                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
                ax_roc.legend(loc='lower right', fontsize=10)
                ax_roc.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_roc)
                plt.close(fig_roc)
            else:
                st.info("ROC curve is only available for binary classification.")
        
        # ---------------------------------------------------------
        # 混淆矩阵
        # ---------------------------------------------------------
        with viz_tab2:
            if "confusion_matrix" in metrics:
                cm = metrics["confusion_matrix"]
                
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                
                # 使用 imshow 绘制热力图
                im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
                
                # 添加颜色条
                cbar = ax_cm.figure.colorbar(im, ax=ax_cm)
                cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=11)
                
                # 设置标签
                classes = ['Normal (0)', 'Anomaly (1)']
                ax_cm.set(xticks=np.arange(cm.shape[1]),
                         yticks=np.arange(cm.shape[0]),
                         xticklabels=classes, yticklabels=classes,
                         ylabel='Actual Label',
                         xlabel='Predicted Label')
                
                ax_cm.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
                
                # 在每个格子中显示数值
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax_cm.text(j, i, format(cm[i, j], 'd'),
                                  ha="center", va="center",
                                  color="white" if cm[i, j] > thresh else "black",
                                  fontsize=20, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)
                
                # 显示混淆矩阵解读
                tn, fp, fn, tp = cm.ravel()
                col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
                col_cm1.metric("True Negative", tn, help="Correctly predicted as Normal")
                col_cm2.metric("False Positive", fp, help="Normal misclassified as Anomaly")
                col_cm3.metric("False Negative", fn, help="Anomaly misclassified as Normal")
                col_cm4.metric("True Positive", tp, help="Correctly predicted as Anomaly")
        
        # ---------------------------------------------------------
        # 特征重要性
        # ---------------------------------------------------------
        with viz_tab3:
            if "feature_importance" in metrics:
                importance = metrics["feature_importance"]
                
                # 排序
                sorted_importance = dict(sorted(importance.items(), 
                                                key=lambda x: x[1], reverse=True))
                
                fig_fi, ax_fi = plt.subplots(figsize=(10, max(6, len(importance) * 0.4)))
                
                features = list(sorted_importance.keys())
                values = list(sorted_importance.values())
                
                # 使用渐变色
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))[::-1]
                
                bars = ax_fi.barh(features[::-1], values[::-1], color=colors)
                
                ax_fi.set_xlabel('Importance Score', fontsize=12)
                ax_fi.set_title('Feature Importance (LightGBM)', fontsize=14, fontweight='bold')
                ax_fi.grid(True, axis='x', alpha=0.3)
                
                # 在条形上显示数值
                for bar, val in zip(bars, values[::-1]):
                    ax_fi.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                              f'{val:.0f}', va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig_fi)
                plt.close(fig_fi)
                
                # 显示 Top 5 特征
                st.markdown("#### 🏆 Top 5 Most Important Features")
                top5 = list(sorted_importance.items())[:5]
                top5_df = pd.DataFrame(top5, columns=['Feature', 'Importance'])
                top5_df['Rank'] = range(1, len(top5_df) + 1)
                top5_df = top5_df[['Rank', 'Feature', 'Importance']]
                st.dataframe(top5_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
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
                    pred_arr, prob_arr = predict_with_threshold(model, sample_data)
                    pred = int(pred_arr[0])
                    prob = float(prob_arr[0])
                    
                    if pred == 1:
                        st.error(f"🚨 ANOMALY DETECTED (Risk Score: {prob:.4f})")
                        
                        # SHAP 解释（使用缓存的 explainer）
                        shap_values = explainer.shap_values(sample_data)
                        vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                        
                        top_indices = np.argsort(vals)[::-1][:3]
                        feat_names = sample_data.columns
                        
                        st.markdown("### 🛠️ Smart Repair Suggestions")
                        
                        # 收集修复建议
                        repair_suggestions = {}
                        for idx in top_indices:
                            if vals[idx] > 0:
                                fname = feat_names[idx]
                                report, _ = st.session_state.repairer.generate_repair_suggestion(sample_data, fname)
                                repair_suggestions[fname] = {
                                    'current': sample_data[fname].values[0],
                                    'suggested': report['Suggested Value'],
                                    'impact': vals[idx],
                                    'logic': report['Repair Logic']
                                }
                                
                                with st.expander(f"🔴 Issue: {fname} (Impact: +{vals[idx]:.2f})", expanded=True):
                                    st.write(f"**Current:** {sample_data[fname].values[0]}")
                                    st.success(f"**Suggested:** {report['Suggested Value']}")
                                    st.caption(f"Reasoning: {report['Repair Logic']}")
                        
                        # 保存原始数据和修复建议到 session_state
                        st.session_state.current_sample = sample_data.copy()
                        st.session_state.repair_suggestions = repair_suggestions
                        st.session_state.original_prob = prob
                        
                    else:
                        st.success(f"✅ Normal Profile (Risk Score: {prob:.4f})")
                        # 清除之前的修复状态
                        if 'repair_suggestions' in st.session_state:
                            del st.session_state.repair_suggestions
                
                # =========================================================
                # 🔄 修复验证功能
                # =========================================================
                if 'repair_suggestions' in st.session_state and st.session_state.repair_suggestions:
                    st.markdown("---")
                    st.markdown("### 🔄 Repair Verification")
                    st.info("Apply the suggested repairs and verify if the sample becomes normal.")
                    
                    if st.button("✨ Apply All Repairs & Verify", key="apply_repairs", type="primary"):
                        # 创建修复后的数据副本
                        repaired_data = st.session_state.current_sample.copy()
                        
                        # 应用所有修复建议
                        for fname, repair_info in st.session_state.repair_suggestions.items():
                            repaired_data[fname] = repair_info['suggested']
                        
                        # 重新预测
                        new_pred_arr, new_prob_arr = predict_with_threshold(model, repaired_data)
                        new_pred = int(new_pred_arr[0])
                        new_prob = float(new_prob_arr[0])
                        original_prob = st.session_state.original_prob
                        
                        # 显示修复前后对比
                        st.markdown("#### 📊 Before vs After Comparison")
                        
                        comparison_data = []
                        for fname, repair_info in st.session_state.repair_suggestions.items():
                            comparison_data.append({
                                'Feature': fname,
                                'Before': repair_info['current'],
                                'After': repair_info['suggested'],
                                'Impact': f"+{repair_info['impact']:.2f}"
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        # 显示预测结果对比
                        st.markdown("#### 🎯 Prediction Result")
                        col_before, col_after, col_change = st.columns(3)
                        
                        with col_before:
                            st.metric(
                                "Before Repair",
                                "🚨 Anomaly",
                                f"Risk: {original_prob:.4f}"
                            )
                        
                        with col_after:
                            if new_pred == 0:
                                st.metric(
                                    "After Repair",
                                    "✅ Normal",
                                    f"Risk: {new_prob:.4f}"
                                )
                            else:
                                st.metric(
                                    "After Repair",
                                    "🚨 Still Anomaly",
                                    f"Risk: {new_prob:.4f}"
                                )
                        
                        with col_change:
                            risk_change = new_prob - original_prob
                            st.metric(
                                "Risk Change",
                                f"{risk_change:+.4f}",
                                f"{risk_change/original_prob*100:+.1f}%" if original_prob > 0 else "N/A",
                                delta_color="inverse"
                            )
                        
                        # 验证结果
                        if new_pred == 0:
                            st.success("🎉 **Repair Successful!** The sample is now classified as Normal.")
                            st.balloons()
                        else:
                            st.warning("⚠️ **Partial Improvement.** The sample is still classified as Anomaly, but risk score decreased. Consider additional repairs.")
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
                predictions, probabilities = predict_with_threshold(model, batch_data)
                
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
