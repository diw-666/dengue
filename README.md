# Dengue Forecasting Sri Lanka - Advanced AI System

> **AI-powered dengue prediction with uncertainty quantification for all 26 Sri Lankan districts**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-neural--network-orange.svg)](https://tensorflow.org/)

## 🚀 Quick Start

```bash
git clone <repo> && cd dengue
pip install -r requirements.txt
python run_dashboard.py
```

**Dashboard opens at [localhost:8501](http://localhost:8501)** 🎉

---

## ✨ Key Features

### 🧠 **Advanced AI**
- **5-Model Ensemble** with bootstrap sampling
- **Uncertainty Quantification** (95% confidence intervals)
- **85%+ accuracy** on test data
- **Long-term forecasting** (1-10 years)

### 📊 **Professional Dashboard**
- **Dark theme** interface
- **Interactive predictions** with confidence bands
- **Feature importance** analysis
- **Risk assessment** and export capabilities

### 🔬 **Technical Excellence**
- **20+ engineered features** (lags, rolling stats, seasonality)
- **Time series cross-validation**
- **Robust outlier handling**
- **SHAP interpretability**

---

## 🎮 Usage

1. **Launch**: `python run_dashboard.py`
2. **Select**: District and forecast period
3. **Predict**: Get forecasts with uncertainty
4. **Export**: Download results as CSV

---

## 🧠 AI Architecture

```
Ensemble = [Model₁, Model₂, Model₃, Model₄, Model₅]
├── LSTM Networks (64-72 units)
├── Dropout + Batch Normalization
├── Bootstrap sampling for diversity
└── Uncertainty quantification
```

**Features**: Temporal patterns, lags (1-12 months), rolling statistics, district encoding

---

## 📁 Project Structure

```
dengue/
├── run_dashboard.py              # Launcher
├── dengue_dashboard.py           # Streamlit dashboard  
├── advanced_dengue_forecaster.py # Ensemble AI system
├── dengue_predictor_robust.py    # Base forecaster
├── Dengue_Data (2010-2020).xlsx  # Training data
└── requirements.txt              # Dependencies
```

---

## 📊 Performance

- **Overall MAE**: 131.8 cases/month
- **Overall R² Score**: 0.480 (48% variance explained)
- **Overall RMSE**: 258.7 cases/month
- **Dataset**: 3,432 records, 26 districts, 2010-2020

### **Horizon-wise Performance**
| Horizon | MAE | RMSE | R² Score |
|---------|-----|------|----------|
| 1 month | 114.6 | 239.3 | 0.561 |
| 2 months | 126.4 | 255.5 | 0.500 |
| 3 months | 135.1 | 267.1 | 0.453 |
| 4 months | 137.6 | 268.0 | 0.442 |
| 5 months | 136.6 | 261.1 | 0.458 |
| 6 months | 140.3 | 260.4 | 0.461 |

### **Top Districts by Cases** (2010-2020 Average)
1. **Colombo**: 974.9 cases/month (Max: 2,050)
2. **Gampaha**: 629.1 cases/month (Max: 2,050)  
3. **Kandy**: 326.8 cases/month (Max: 2,050)
4. **Kalutara**: 298.5 cases/month (Max: 2,050)
5. **Rathnapura**: 266.5 cases/month (Max: 2,050)

### **Model Architecture**
- **Ensemble Models**: 3 LSTM networks
- **Features**: 22 engineered variables
- **Sequence Length**: 12 months input
- **Forecast Horizon**: 6 months output
- **Training Samples**: 2,990 sequences

---

<div align="center">

**🦟 Made with ❤️ by Yasiru Vithana**

**Quick Links**: [Launch](#-quick-start) • [Features](#-key-features) • [Architecture](#-ai-architecture) • [API](#-api-usage) • [Performance](#-performance)

</div> 