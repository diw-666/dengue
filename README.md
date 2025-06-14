# Dengue Forecasting Sri Lanka - Advanced AI System

> **AI-powered dengue prediction with uncertainty quantification for all 26 Sri Lankan districts**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-neural--network-orange.svg)](https://tensorflow.org/)

## 🚀 Quick Start

```bash
git clone diw-666 && cd dengue
pip install -r requirements.txt
python run_dashboard.py
```

**Dashboard opens at [localhost:8501](http://localhost:8501)** 🎉

---

## ✨ Key Features

### 🧠 **Advanced AI Architecture**
- **3-Model Ensemble** with bootstrap sampling
- **LSTM Networks** with attention mechanisms
- **22 engineered features** (temporal patterns, lags, rolling statistics)
- **Time series cross-validation** for robust evaluation

### 📊 **Professional Dashboard**
- **Dark theme** modern interface
- **Interactive predictions** with Plotly visualizations
- **Uncertainty quantification** with confidence intervals
- **Risk assessment** and trend analysis
- **CSV export** capabilities

### 🔬 **Scientific Approach**
- **Proper temporal validation** (no data leakage)
- **Feature importance** analysis with SHAP
- **Bootstrap confidence intervals**
- **Comprehensive evaluation metrics**
- **Long-term forecasting** up to 10 years

---

## 🎮 Usage

### **Dashboard Interface**
1. **Launch**: `python run_dashboard.py`
2. **Select**: Choose district from 26 options
3. **Configure**: Set forecast period (1-10 years)
4. **Predict**: Get forecasts with uncertainty bands
5. **Export**: Download results as CSV

---

## 🧠 AI Architecture

### **Ensemble System**
```
Advanced Forecaster
├── Base LSTM Model (64 units)
├── Ensemble Models (3x LSTM variants)
├── Bootstrap Sampling
├── Uncertainty Quantification
└── Feature Importance Analysis
```

### **Feature Engineering**
- **Temporal Features**: Month, quarter, year encoding
- **Lag Features**: 1, 2, 3, 6, 12-month historical values
- **Rolling Statistics**: 3, 6, 12-month means and standard deviations
- **Trend Features**: First and seasonal differences
- **District Encoding**: Normalized location-specific patterns

### **Training Process**
1. **Data Preparation**: 3,432 records from 26 districts (2010-2020)
2. **Sequence Creation**: 2,990 training samples with 12-month input
3. **Time Series Splitting**: Proper temporal validation
4. **Ensemble Training**: 3 models with different architectures
5. **Uncertainty Calibration**: Bootstrap prediction intervals

---

## 📊 Performance

### **Actual Test Results** (Time Series Cross-Validation)
- **Overall MAE**: 131.8 cases/month
- **Overall R² Score**: 0.480 (48% variance explained)
- **Overall RMSE**: 258.7 cases/month
- **Dataset**: 3,432 records, 26 districts, 2010-2020

### **Horizon-wise Performance**
| Forecast Period | MAE | RMSE | R² Score | Performance |
|----------------|-----|------|----------|-------------|
| 1 month ahead | 114.6 | 239.3 | 0.561 | **Best** |
| 2 months ahead | 126.4 | 255.5 | 0.500 | Good |
| 3 months ahead | 135.1 | 267.1 | 0.453 | Fair |
| 4 months ahead | 137.6 | 268.0 | 0.442 | Fair |
| 5 months ahead | 136.6 | 261.1 | 0.458 | Fair |
| 6 months ahead | 140.3 | 260.4 | 0.461 | Fair |

### **District Performance** (Historical Averages 2010-2020)
| Rank | District | Avg Cases/Month | Peak Cases | Data Quality |
|------|----------|----------------|------------|--------------|
| 1 | **Colombo** | 974.9 | 2,050 | Excellent |
| 2 | **Gampaha** | 629.1 | 2,050 | Excellent |
| 3 | **Kandy** | 326.8 | 2,050 | Excellent |
| 4 | **Kalutara** | 298.5 | 2,050 | Excellent |
| 5 | **Rathnapura** | 266.5 | 2,050 | Excellent |

*All 26 districts have 132 monthly records each*

---

## 📁 Project Structure

```
dengue/
├── 🚀 run_dashboard.py              # Smart launcher
├── 📊 dengue_dashboard.py           # Streamlit dashboard  
├── 🧠 advanced_dengue_forecaster.py # Ensemble AI system
├── 🔧 dengue_predictor_robust.py    # Base forecaster
├── 📈 Dengue_Data (2010-2020).xlsx  # Training data (3,432 records)
├── 📦 advanced_dengue_forecaster.pkl # Trained models
├── 🤖 robust_dengue_model.h5        # Neural network weights
├── 📋 requirements.txt              # Dependencies
├── ⚙️ pyproject.toml               # Project config
└── 🌐 runtime.txt                  # Deployment config
```

---

## ⚠️ Limitations & Disclaimers

### **Model Limitations**
- **Accuracy**: 48% variance explained - moderate predictive power
- **Horizon Degradation**: Performance decreases for longer forecasts
- **Data Dependency**: Based on 2010-2020 historical patterns
- **External Factors**: Doesn't include weather, population, or policy changes

### **Important Notes**
- **Research Tool**: For planning and analysis purposes only
- **Medical Decisions**: Always consult healthcare professionals
- **Validation**: Continuously validate with new data
- **Uncertainty**: Consider confidence intervals in all decisions
- **Updates**: Model may need retraining with recent data

---

## 🛠️ Development

### **Local Setup**
```bash
# Clone repository
git clone diw-666/dengue 
cd dengue

# Install dependencies
pip install -r requirements.txt

# Train models (optional - auto-trained on first run)
python advanced_dengue_forecaster.py

# Launch dashboard
python run_dashboard.py
```

## 🤝 Contributing

### **Areas for Improvement**
- **Weather Integration**: Add meteorological data
- **Real-time Data**: Connect to live surveillance systems
- **Mobile Interface**: Responsive design improvements
- **Additional Models**: XGBoost, Prophet integration
- **Multi-country**: Expand to other dengue-endemic regions

### **How to Contribute**
1. **Fork** the repository
2. **Create** feature branch
3. **Implement** improvements
4. **Test** thoroughly
5. **Submit** pull request

---

**🦟 Made with ❤️ by Yasiru Vithana**

*Advanced AI for dengue prevention in Sri Lanka*

⭐ **Star this repo if it's helping protect communities!**

---

**🔗 Links**: [Quick Start](#-quick-start) • [Features](#-key-features) • [Performance](#-performance) • [Architecture](#-ai-architecture) • [Usage](#-usage)

</div> 