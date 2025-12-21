# Crisis Radar: ML Pipeline for S&P 500 Drawdown Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](https://github.com)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-orange.svg)](https://crisis-radar-finclub.streamlit.app/)

A production-ready time-series machine learning pipeline that predicts the probability of large drawdowns in the S&P 500 index over the next N trading days. Features a user-friendly Streamlit web interface for model training, live predictions, and comprehensive evaluation.

## 🌐 Live Demo

**Try it online:** [https://crisis-radar-finclub.streamlit.app/](https://crisis-radar-finclub.streamlit.app/)

The application is deployed on Streamlit Cloud and ready to use. No installation required!

## 🌟 Features

- **🤖 Multiple ML Models**: Random Forest, Gradient Boosting, and Logistic Regression
- **📊 Interactive Web GUI**: Streamlit-based interface for easy interaction
- **📈 Live Predictions**: Real-time risk scores using latest market data
- **🎯 Probability Calibration**: Well-calibrated probabilities for accurate risk assessment
- **📉 Comprehensive Metrics**: PR-AUC, ROC-AUC, Brier score, precision/recall
- **📁 Custom Test Data**: Upload and test your own datasets
- **⏱️ Time-Series Correct**: No data leakage, strict temporal splits

## 📖 Overview

Crisis Radar predicts the probability that the S&P 500 will experience a drawdown of at least **-8%** (default) within the next **20 trading days** (default). The system uses historical market data and volatility indicators to forecast potential crisis events.

### Key Capabilities

- **Binary Classification**: Predicts crisis events (drawdown ≥ threshold)
- **Probability Output**: Provides calibrated probabilities (0-100%)
- **Real-Time Analysis**: Uses latest market data from yfinance
- **Comprehensive Evaluation**: Multiple metrics for model assessment

## 🚀 Quick Start

### Option 1: Use Live Demo (Recommended)

**No installation needed!** Try the application online:

👉 **[https://crisis-radar-finclub.streamlit.app/](https://crisis-radar-finclub.streamlit.app/)**

The application is fully functional and ready to use. Just open the link and start training models!

### Option 2: Run Locally

#### Prerequisites

- Python 3.8 or higher
- pip package manager

#### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

#### Run the Pipeline

**Using GUI (Recommended)**

```bash
streamlit run streamlit_app.py
```

The GUI will open at `http://localhost:8501`

**Important:** On first run, you need to train a model:
1. Go to **"Train Model"** page in the GUI
2. Configure settings (or use defaults)
3. Click **"Train Model"**
4. Wait for training to complete (may take a few minutes)
5. After training, you can use **"Live Predictions"** and other features

**Using Command Line**

```bash
# Train model with default settings (required first time)
python -m crisis_radar.cli run

# Get live risk score (after training)
python -m crisis_radar.cli live
```

## 💻 GUI Usage

The Streamlit GUI provides an intuitive web interface with the following pages:

1. **🏠 Home**: Overview dashboard and status
2. **🚀 Train Model**: Configure and train ML models with interactive forms
3. **📈 Live Predictions**: Get real-time risk scores with latest market data
4. **📊 Test Data**: Upload custom CSV files for testing
5. **📉 Visualizations**: View performance metrics, calibration curves, and feature importance
6. **ℹ️ About**: Project information and documentation

### Test Data Format

When uploading test data, your CSV should have:
- **Date** column (as index): YYYY-MM-DD format
- **GSPC_Close**: S&P 500 closing prices
- **VIX_Close**: VIX closing prices

Example:
```csv
Date,GSPC_Close,VIX_Close
2024-01-01,4500.0,15.5
2024-01-02,4510.0,16.0
```

## ⌨️ Command Line Usage

### Train Model

Train a model with custom parameters:

```bash
python -m crisis_radar.cli run \
    --horizon 20 \
    --dd 0.08 \
    --train-start 2005-01-01 \
    --train-end 2015-12-31 \
    --test-start 2016-01-01 \
    --model rf \
    --calibration isotonic
```

**Available Options:**
- `--horizon`: Prediction horizon in trading days (default: 20)
- `--dd`: Drawdown threshold as fraction (default: 0.08 for -8%)
- `--train-start`, `--train-end`: Training period dates
- `--test-start`, `--test-end`: Test period dates
- `--model`: Model type (`rf`, `gb`, or `logreg`)
- `--calibration`: Calibration method (`isotonic` or `sigmoid`)

### Get Live Predictions

```bash
# With cached data (faster)
python -m crisis_radar.cli live

# With fresh data download (slower but more accurate)
python -m crisis_radar.cli live --refresh
```

## What You Get

After training a model, these files are generated locally:

- **Trained Model**: `artifacts/models/calibrated_model.pkl` (saved locally)
- **Metrics**: `artifacts/metrics/test_metrics.json` (saved locally)
- **Predictions**: `artifacts/metrics/test_predictions.csv` (saved locally)
- **Visualizations**: `reports/figures/*.png` (saved locally)

**Note:** These files are **not included in the GitHub repository** (they are gitignored). Each user trains their own model when they first run the application. This keeps the repository lightweight and ensures everyone uses fresh, up-to-date models.

## 📦 Dependencies

The project requires the following Python packages:

- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.23.0` - Numerical computing
- `scikit-learn>=1.2.0` - Machine learning
- `matplotlib>=3.6.0` - Visualization
- `yfinance>=0.2.0` - Market data
- `streamlit>=1.28.0` - Web interface
- `scipy>=1.10.0` - Scientific computing

All dependencies are listed in `requirements.txt` and will be installed automatically.

## Project Structure

```
crisis-radar/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── pyproject.toml         # Package config
├── streamlit_app.py       # GUI application
├── src/crisis_radar/      # Python ML code
├── scripts/               # Utility scripts
└── .streamlit/           # Streamlit config
```

## 🔧 How It Works

The pipeline follows these steps:

1. **📥 Data Download**: Downloads S&P 500 (^GSPC) and VIX (^VIX) data using yfinance
2. **🔨 Feature Engineering**: Builds features from historical data:
   - Return features (1-day, 5-day, 20-day)
   - Volatility features (realized volatility)
   - Momentum features (SMA ratios)
   - Drawdown features
   - VIX indicators
   - Distribution features (skewness, kurtosis)
3. **🏷️ Label Creation**: Creates binary labels for future drawdown events
4. **🎓 Model Training**: Trains selected ML model with time-series cross-validation
5. **📊 Probability Calibration**: Calibrates probabilities using Isotonic or Sigmoid regression
6. **✅ Evaluation**: Evaluates on out-of-sample test set with comprehensive metrics

## 📊 Evaluation Metrics

The pipeline computes comprehensive metrics:

- **PR-AUC**: Precision-Recall Area Under Curve (primary metric for imbalanced data)
- **ROC-AUC**: Receiver Operating Characteristic AUC
- **Brier Score**: Calibration metric (lower is better)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: At optimal threshold (maximizes F1)

## ⚠️ Limitations & Disclaimers

**Important:** This tool is for educational and research purposes only. Not financial advice.

- **Regime Assumptions**: Model assumes relatively stable market regimes
- **No Transaction Costs**: Does not account for trading costs, slippage, or market impact
- **Binary Classification**: Predicts crisis/no-crisis, not magnitude or timing
- **Single Asset**: Focuses on S&P 500 only (no cross-asset relationships)
- **No Online Learning**: Model is trained once and not updated automatically
- **Historical Data**: Past performance does not guarantee future results

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is provided as-is for educational and research purposes.

## 👤 Author

**Apostolos Chardalias** | Finance Club UniPi

## 🌐 Deployment

The application is deployed on **Streamlit Cloud**:

**🔗 Live URL:** [https://crisis-radar-finclub.streamlit.app/](https://crisis-radar-finclub.streamlit.app/)

### Deploy Your Own

To deploy on Streamlit Cloud:

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `streamlit_app.py` as the main file
5. Deploy!

## 🙏 Acknowledgments

- Finance Club UniPi for support
- yfinance for market data access
- Streamlit for the web framework
- scikit-learn for ML algorithms

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

⭐ If you find this project useful, please consider giving it a star!
