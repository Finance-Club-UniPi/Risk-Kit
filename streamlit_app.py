"""Streamlit GUI for Crisis Radar ML Pipeline."""

import io
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from datetime import date, datetime

# Import crisis_radar modules
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from crisis_radar.config import CrisisRadarConfig
from crisis_radar.live import get_live_risk_score
from crisis_radar.pipeline import run_pipeline
from crisis_radar.utils import load_json, setup_logging
from crisis_radar.data import download_market_data
from crisis_radar.features import build_features, get_feature_names
from crisis_radar.evaluation import evaluate_model
from crisis_radar.plotting import (
    plot_probability_timeline,
    plot_precision_recall_curve,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
)

# Page config
st.set_page_config(
    page_title="Crisis Radar - ML Pipeline",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# Header
st.markdown('<div class="main-header">Crisis Radar - ML Pipeline</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["Home", "Train Model", "Live Predictions", "Test Data", "Visualizations", "About"],
    )
    st.markdown("---")
    
    # Check if model exists
    model_path = Path("artifacts/models/calibrated_model.pkl")
    if model_path.exists():
        st.success("Model trained and ready")
        st.session_state.model_trained = True
    else:
        st.warning("No trained model found. Train a model first.")
        st.session_state.model_trained = False
    
    st.markdown("---")
    st.markdown("### Credits")
    st.markdown("**Developed by:** Apostolos Chardalias | Finance Club UniPi")
    st.markdown("**Version:** 1.0.0")

# Home Page
if page == "Home":
    st.header("Welcome to Crisis Radar")
    st.markdown("""
    **Crisis Radar** is a machine learning pipeline that predicts the probability of large drawdowns 
    in the S&P 500 index over the next N trading days.
    
    ### Features:
    - **Train ML Models**: Train Random Forest, Gradient Boosting, or Logistic Regression models
    - **Live Predictions**: Get real-time risk scores using latest market data
    - **Test Data**: Upload your own test data for evaluation
    - **Visualizations**: View performance metrics, calibration curves, and feature importance
    
    ### Quick Start:
    1. Go to **Train Model** to train your first model
    2. Use **Live Predictions** to get current risk scores
    3. Upload test data in **Test Data** for custom evaluation
    4. View results in **Visualizations**
    """)
    
    st.markdown("---")
    st.markdown("**Developed by:** Apostolos Chardalias | Finance Club UniPi")
    
    # Show current status
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.model_trained:
            st.success("Model Ready")
        else:
            st.error("❌ No Model")
    
    with col2:
        if Path("artifacts/metrics/test_metrics.json").exists():
            metrics = load_json("artifacts/metrics/test_metrics.json")
            st.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
        else:
            st.metric("PR-AUC", "N/A")
    
    with col3:
        if Path("data/raw/market_data.csv").exists():
            st.success("Data Cached")
        else:
            st.info("No Cached Data")

# Train Model Page
elif page == "Train Model":
    st.header("Train ML Model")
    
    with st.expander("📖 Instructions", expanded=False):
        st.markdown("""
        This page allows you to train a machine learning model to predict S&P 500 drawdown events.
        
        **Steps:**
        1. Configure model parameters
        2. Set training/test date ranges
        3. Click "Train Model" to start training
        4. View results and metrics after training completes
        """)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        model_type = st.selectbox("Model Type", ["rf", "gb", "logreg"], index=0)
        calibration_method = st.selectbox("Calibration Method", ["isotonic", "sigmoid"], index=0)
        horizon_days = st.number_input("Prediction Horizon (days)", min_value=5, max_value=60, value=20)
        drawdown_threshold = st.slider("Drawdown Threshold (%)", min_value=1.0, max_value=20.0, value=8.0) / 100
    
    with col2:
        st.subheader("Data Configuration")
        train_start = st.date_input("Training Start Date", value=datetime(2005, 1, 1).date())
        train_end = st.date_input("Training End Date", value=datetime(2015, 12, 31).date())
        test_start = st.date_input("Test Start Date", value=datetime(2016, 1, 1).date())
        test_end = st.date_input("Test End Date", value=date.today())
        
        data_start = st.date_input("Data Download Start", value=datetime(2000, 1, 1).date())
        random_seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        n_estimators = st.number_input("N Estimators (RF/GB)", min_value=10, max_value=1000, value=100)
        max_depth = st.number_input("Max Depth (RF/GB)", min_value=1, max_value=20, value=None, help="None = unlimited")
    
    # Train button
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model... This may take a few minutes."):
            try:
                config = CrisisRadarConfig(
                    start_date=data_start.strftime("%Y-%m-%d"),
                    end_date=test_end.strftime("%Y-%m-%d"),
                    horizon_days=int(horizon_days),
                    drawdown_threshold=float(drawdown_threshold),
                    train_start=train_start.strftime("%Y-%m-%d"),
                    train_end=train_end.strftime("%Y-%m-%d"),
                    test_start=test_start.strftime("%Y-%m-%d"),
                    test_end=test_end.strftime("%Y-%m-%d"),
                    model_type=model_type,
                    calibration_method=calibration_method,
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth) if max_depth else None,
                    random_seed=int(random_seed),
                )
                
                artifacts = run_pipeline(config)
                st.session_state.pipeline_run = True
                st.session_state.model_trained = True
                
                st.success("Model trained successfully!")
                st.balloons()
                
                # Show metrics
                if Path(artifacts["metrics"]).exists():
                    metrics = load_json(artifacts["metrics"])
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
                    with col2:
                        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                    with col3:
                        st.metric("Brier Score", f"{metrics['brier_score']:.4f}")
                    with col4:
                        st.metric("F1 Score", f"{metrics['f1']:.4f}")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)

# Live Predictions Page
elif page == "Live Predictions":
    st.header("Live Risk Score Predictions")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first in the 'Train Model' page.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Get Live Prediction")
            force_refresh = st.checkbox("Force Fresh Data Download", value=False, 
                                       help="Download latest data from yfinance (slower but more accurate)")
        
        with col2:
            st.subheader("Model Info")
            if Path("artifacts/configs/pipeline_config.json").exists():
                config = load_json("artifacts/configs/pipeline_config.json")
                st.info(f"Horizon: {config['horizon_days']} days\nThreshold: -{config['drawdown_threshold']:.1%}")
        
        if st.button("Get Live Risk Score", type="primary"):
            with st.spinner("Fetching latest data and computing risk score..."):
                try:
                    result = get_live_risk_score(force_refresh=force_refresh)
                    
                    # Display result
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        st.markdown(f"### Date: {result['date']}")
                        st.markdown(f"**Prediction Horizon:** {result['horizon_days']} trading days")
                        st.markdown(f"**Drawdown Threshold:** -{result['drawdown_threshold']:.1%}")
                        
                        st.markdown("---")
                        
                        # Risk level with color
                        risk_class = f"risk-{result['risk_level'].lower()}"
                        st.markdown(f'<div class="{risk_class}">Risk Level: {result["risk_level"]}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Probability gauge
                        prob = result['probability']
                        st.metric("Crisis Probability", f"{prob:.2%}")
                        
                        # Progress bar
                        st.progress(float(prob))
                        
                        # Interpretation
                        st.markdown("---")
                        st.info(f"""
                        **Interpretation:**
                        
                        There is a **{prob:.1%}** probability that the S&P 500 will experience 
                        a drawdown of at least **{result['drawdown_threshold']:.1%}** within 
                        the next **{result['horizon_days']}** trading days.
                        """)
                
                except Exception as e:
                    st.error(f"❌ Failed to get live risk score: {str(e)}")
                    st.exception(e)

# Test Data Page
elif page == "Test Data":
    st.header("Upload and Test Custom Data")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first in the 'Train Model' page.")
    else:
        st.markdown("""
        Upload your own test data in CSV format. The file should contain:
        - **Date column**: Date index (YYYY-MM-DD format)
        - **GSPC_Close**: S&P 500 closing prices
        - **VIX_Close**: VIX closing prices
        """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df_uploaded = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                
                st.success(f"File uploaded successfully! ({len(df_uploaded)} rows)")
                
                # Show preview
                with st.expander("Data Preview"):
                    st.dataframe(df_uploaded.head(10))
                    st.write(f"Shape: {df_uploaded.shape}")
                    st.write(f"Columns: {list(df_uploaded.columns)}")
                
                # Check required columns
                required_cols = ["GSPC_Close", "VIX_Close"]
                missing_cols = [col for col in required_cols if col not in df_uploaded.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    if st.button("Test on Uploaded Data", type="primary"):
                        with st.spinner("Processing test data..."):
                            try:
                                # Build features
                                df_spx = df_uploaded["GSPC_Close"]
                                df_vix = df_uploaded["VIX_Close"]
                                df_features = build_features(df_spx, df_vix)
                                
                                # Load model
                                with open("artifacts/models/calibrated_model.pkl", "rb") as f:
                                    model = pickle.load(f)
                                
                                # Get predictions
                                feature_names = get_feature_names()
                                X_test = df_features[feature_names].values
                                
                                # Check for NaN
                                if pd.isna(X_test).any().any():
                                    st.warning("Some features contain NaN values. Filling with forward-fill...")
                                    X_test = pd.DataFrame(X_test, columns=feature_names).ffill().values
                                
                                y_pred_proba = model.predict_proba(X_test)[:, 1]
                                
                                # Create results dataframe
                                results_df = pd.DataFrame({
                                    "date": df_features.index,
                                    "predicted_probability": y_pred_proba,
                                })
                                
                                st.success("Predictions computed!")
                                
                                # Show results
                                st.subheader("Prediction Results")
                                st.dataframe(results_df)
                                
                                # Download button
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Predictions CSV",
                                    data=csv,
                                    file_name=f"predictions_{date.today().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                )
                                
                                # Statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mean Probability", f"{y_pred_proba.mean():.2%}")
                                with col2:
                                    st.metric("Max Probability", f"{y_pred_proba.max():.2%}")
                                with col3:
                                    high_risk = (y_pred_proba >= 0.3).sum()
                                    st.metric("High Risk Days", f"{high_risk} ({high_risk/len(y_pred_proba):.1%})")
                            
                            except Exception as e:
                                st.error(f"❌ Testing failed: {str(e)}")
                                st.exception(e)
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.exception(e)

# Visualizations Page
elif page == "Visualizations":
    st.header("Model Performance Visualizations")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first to view visualizations.")
    else:
        # Check if figures exist
        figures_dir = Path("reports/figures")
        
        if not figures_dir.exists() or len(list(figures_dir.glob("*.png"))) == 0:
            st.info("No visualizations found. Train a model first to generate plots.")
        else:
            # Load metrics
            metrics_path = Path("artifacts/metrics/test_metrics.json")
            if metrics_path.exists():
                metrics = load_json(str(metrics_path))
                
                # Metrics summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
                with col2:
                    st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                with col3:
                    st.metric("Brier Score", f"{metrics['brier_score']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1']:.4f}")
                
                st.markdown("---")
            
            # Display figures
            fig_files = {
                "Probability Timeline": "probability_timeline.png",
                "Precision-Recall Curve": "precision_recall_curve.png",
                "Calibration Curve": "calibration_curve.png",
                "Confusion Matrix": "confusion_matrix.png",
                "Feature Importance": "feature_importance.png",
            }
            
            for fig_name, fig_file in fig_files.items():
                fig_path = figures_dir / fig_file
                if fig_path.exists():
                    st.subheader(fig_name)
                    st.image(str(fig_path))
                    st.markdown("---")

# About Page
elif page == "About":
    st.header("About Crisis Radar")
    
    st.markdown("""
    ### Overview
    Crisis Radar is a machine learning pipeline for predicting large drawdown events in the S&P 500 index.
    
    ### Features
    - **Time-series ML models**: Random Forest, Gradient Boosting, Logistic Regression
    - **Probability calibration**: Isotonic or Sigmoid calibration
    - **Comprehensive evaluation**: PR-AUC, ROC-AUC, Brier score, precision/recall
    - **Real-time predictions**: Live risk scores using latest market data
    - **Custom test data**: Upload and test your own datasets
    
    ### Methodology
    - **Target**: Binary classification of drawdown events (default: -8% within 20 trading days)
    - **Features**: Returns, volatility, momentum, drawdown, VIX indicators
    - **Validation**: Strict temporal split (no data leakage)
    - **Calibration**: TimeSeriesSplit cross-validation for probability calibration
    
    ### Data Sources
    - S&P 500 Index (^GSPC) via yfinance
    - VIX Volatility Index (^VIX) via yfinance
    
    ### Project Structure
    ```
    crisis-radar/
    ├── src/crisis_radar/    # Source code
    ├── artifacts/            # Trained models, metrics
    ├── reports/              # Visualizations, summaries
    ├── data/                 # Market data cache
    └── streamlit_app.py      # This GUI application
    ```
    
    ### Documentation
    - See `README.md` for full documentation
    """)
    
    st.markdown("---")
    
    st.markdown("### Credits")
    st.markdown("""
    **Developed by:** Apostolos Chardalias | Finance Club UniPi
    
    **Version:** 1.0.0
    
    **License:** Provided as-is for educational and research purposes.
    """)

