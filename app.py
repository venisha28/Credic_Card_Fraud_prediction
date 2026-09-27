import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure local imports work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import (
    load_model, load_scaler, load_metrics,
    predict_single_transaction, predict_batch, get_risk_level
)
from src.evaluation import (
    plot_confusion_matrix_fig,
    plot_metrics_comparison_fig,
    plot_fraud_distribution_fig
)

# App configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection ML Pipeline",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Dark Theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    .main p, .main span, .main label, .main div {
        color: #F8FAFC !important;
    }
    
    h1, h2, h3, h4 {
        color: #60A5FA !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid #334155 !important;
    }
    
    .sidebar-header {
        font-size: 20px;
        font-weight: bold;
        color: #60A5FA;
        text-align: center;
        padding-bottom: 12px;
        border-bottom: 1px solid #334155;
        margin-bottom: 20px;
    }
    
    .kpi-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .kpi-title {
        font-size: 13px;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #3B82F6 !important;
    }
    
    .stButton>button {
        background-color: #3B82F6 !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #60A5FA !important;
    }
</style>
""", unsafe_allow_html=True)

# Load Model Artifacts safely
@st.cache_resource
def get_artifacts():
    try:
        model = load_model()
        scaler = load_scaler()
        metrics = load_metrics()
        return model, scaler, metrics, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, metrics_data, error_msg = get_artifacts()

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-header">💳 Fraud Detector ML</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔍 Fraud Prediction", "📁 Batch Prediction"],
        index=0
    )
    st.markdown("---")
    st.markdown("### 📌 ML Pipeline Summary")
    st.markdown("""
    - **Models**: Logistic Regression, Random Forest, XGBoost
    - **Resampling**: SMOTE (Training split only)
    - **Dataset**: Kaggle Credit Card Fraud
    """)

if error_msg:
    st.error(f"⚠️ Model artifacts not found: {error_msg}")
    st.info("Please execute `python src/train.py` in your terminal to train and save the model.")
    st.stop()

best_model_name = metrics_data.get('best_model_name', 'Best Model')
all_metrics = metrics_data.get('metrics', {})

# ==============================================================================
# A. DASHBOARD PAGE
# ==============================================================================
if page == "📊 Dashboard":
    st.title("📊 Credit Card Fraud Analytics Dashboard")
    st.markdown("Real-time Machine Learning model evaluation and dataset performance summary.")
    
    # Extract test set metrics for best model
    best_metrics = all_metrics.get(best_model_name, {})
    cm = np.array(best_metrics.get('confusion_matrix', [[0, 0], [0, 0]]))
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    total_test = tn + fp + fn + tp
    total_fraud = fn + tp
    total_legit = tn + fp
    fraud_rate = (total_fraud / total_test * 100) if total_test > 0 else 0.0
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Analyzed</div>
            <div class="kpi-value">{total_test:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Fraudulent Cases</div>
            <div class="kpi-value" style="color: #EF4444 !important;">{total_fraud:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Legitimate Cases</div>
            <div class="kpi-value" style="color: #10B981 !important;">{total_legit:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Fraud Rate</div>
            <div class="kpi-value" style="color: #F59E0B !important;">{fraud_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 2-3 Useful Visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("1. Class Distribution (Test Set)")
        fig_dist = plot_fraud_distribution_fig(total_legit, total_fraud)
        st.pyplot(fig_dist)
        
    with col_chart2:
        st.subheader(f"2. Confusion Matrix ({best_model_name})")
        fig_cm = plot_confusion_matrix_fig(cm, title=f"Confusion Matrix ({best_model_name})")
        st.pyplot(fig_cm)
        
    st.markdown("---")
    st.subheader("3. Model Performance Comparison")
    fig_comp = plot_metrics_comparison_fig(all_metrics)
    st.pyplot(fig_comp)
    
    st.markdown("#### 📈 Best Model Test Metrics")
    metrics_df = pd.DataFrame(all_metrics).T
    st.dataframe(metrics_df[['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']], use_container_width=True)

# ==============================================================================
# B. FRAUD PREDICTION (SINGLE)
# ==============================================================================
elif page == "🔍 Fraud Prediction":
    st.title("🔍 Single Transaction Fraud Prediction")
    st.markdown("Enter transaction feature details to predict fraud probability.")
    
    with st.form("single_pred_form"):
        st.subheader("Transaction Parameters")
        c1, c2 = st.columns(2)
        with c1:
            time_val = st.number_input("Time (seconds from start)", min_value=0.0, value=85000.0, step=100.0)
        with c2:
            amount_val = st.number_input("Amount ($)", min_value=0.0, value=149.99, step=10.0)
            
        st.subheader("PCA Features (V1 to V28)")
        st.caption("PCA features are pre-filled with standard values. You can adjust them for testing.")
        
        v_inputs = {}
        # Display V features in a 4-column layout for clean design
        v_cols_ui = st.columns(4)
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            with v_cols_ui[col_idx]:
                # Give interesting defaults for key fraud features V14, V12, V10, V17 if user tests
                v_inputs[f'V{i}'] = st.number_input(f"V{i}", value=0.0, step=0.1, key=f"v_{i}")
                
        submit_btn = st.form_submit_button("Predict Fraud Risk")
        
    if submit_btn:
        input_dict = {'Time': time_val, 'Amount': amount_val}
        input_dict.update(v_inputs)
        
        res = predict_single_transaction(input_dict, model=model, scaler=scaler)
        prob_pct = res['fraud_probability'] * 100
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Fraud Probability", f"{prob_pct:.1f}%")
        with res_col2:
            st.metric("Prediction", res['prediction'])
        with res_col3:
            st.metric("Risk Level", res['risk_level'])
            
        if res['prediction_code'] == 1:
            st.error("🚨 WARNING: High probability of fraudulent activity detected for this transaction!")
        else:
            st.success("✅ APPROVED: This transaction appears legitimate.")

# ==============================================================================
# C. BATCH PREDICTION
# ==============================================================================
elif page == "📁 Batch Prediction":
    st.title("📁 CSV Batch Fraud Prediction")
    st.markdown("Upload a batch CSV file of transactions to evaluate fraud risk in bulk.")
    
    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"📂 Loaded dataset with **{len(df_upload)}** rows and **{len(df_upload.columns)}** columns.")
            st.dataframe(df_upload.head(3), use_container_width=True)
            
            if st.button("Run Batch Fraud Predictions"):
                with st.spinner("Analyzing transactions..."):
                    df_res = predict_batch(df_upload, model=model, scaler=scaler)
                    
                st.success("Batch Prediction Complete!")
                
                # Summary metrics
                batch_fraud_cnt = sum(df_res['Prediction'] == 'Fraudulent')
                batch_legit_cnt = sum(df_res['Prediction'] == 'Legitimate')
                batch_rate = (batch_fraud_cnt / len(df_res)) * 100
                
                bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                bcol1.metric("Total Batch Rows", len(df_res))
                bcol2.metric("Predicted Fraud", batch_fraud_cnt)
                bcol3.metric("Predicted Legitimate", batch_legit_cnt)
                bcol4.metric("Batch Fraud Rate", f"{batch_rate:.2f}%")
                
                st.markdown("---")
                st.subheader("Prediction Results Table")
                st.dataframe(df_res, use_container_width=True)
                
                # Download CSV
                csv_bytes = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_bytes,
                    file_name="fraud_predictions_result.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
    else:
        st.info("Please upload a CSV file containing `Time`, `Amount`, and `V1`..`V28` features.")
