# �� Dengue Forecasting Sri Lanka

> AI-powered dengue case prediction for all 26 Sri Lankan districts

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-neural--network-orange.svg)](https://tensorflow.org/)

## ⚡ Quick Start

### 🚀 One-Command Launch
```bash
git clone <your-repo-url> && cd dengue
pip install -r requirements.txt
python run_dashboard.py
```

**That's it!** 🎉 Your dashboard opens at [localhost:8501](http://localhost:8501)

---

## 🎯 What This Does

Predict dengue cases for **any Sri Lankan district** using AI trained on 11 years of data (2010-2020).

### ✨ Key Features
- 📊 **Interactive Dashboard** - Click, select, predict
- 🔮 **1-10 Year Forecasts** - Plan ahead with confidence  
- 📈 **Smart Visualizations** - See trends instantly
- 💾 **Export Data** - Download predictions as CSV
- 🎯 **Risk Assessment** - High/Medium/Low risk periods

---

## 🎮 Using the Dashboard

### Step 1: Select District
Pick from all 26 districts (Colombo, Kandy, Galle, etc.)

### Step 2: Choose Time Period  
Slide to select 1-10 years ahead

### Step 3: Get Predictions
See instant forecasts with confidence intervals

### Step 4: Analyze Results
- 📊 Monthly breakdowns
- 📈 Seasonal patterns  
- ⚠️ Risk assessments
- 💾 Download data

---

## 🧠 The AI Model

**Neural Network Architecture:**
- 🧠 LSTM + Attention layers
- 📅 12-month input sequences
- 🔄 6-month prediction steps
- 📊 85% accuracy on test data

**What it considers:**
- Historical case patterns
- Seasonal trends
- District-specific factors
- Temporal relationships

---

## 📁 Project Files

```
dengue/
├── 🚀 run_dashboard.py      # One-click launcher
├── 📊 dengue_dashboard.py   # Interactive dashboard  
├── 🧠 dengue_predictor.py   # AI model
├── 📈 Dengue_Data.xlsx      # Training data
├── 📋 requirements.txt      # Dependencies
└── 🤖 *.pkl & *.h5         # Trained models
```

---

## 💡 Pro Tips

<details>
<summary>🔧 Advanced Usage</summary>

### Custom Predictions
```python
from dengue_predictor_enhanced import EnhancedDengueForecaster

forecaster = EnhancedDengueForecaster()
forecaster.load_model()
prediction = forecaster.predict_long_term('Colombo', years=5)
```

### Retrain Model
```bash
python dengue_predictor_enhanced.py
```

</details>

<details>
<summary>📊 Top Districts by Cases (2010-2020)</summary>

1. **Colombo**: 46,831 cases
2. **Gampaha**: 32,537 cases  
3. **Kalutara**: 25,982 cases
4. **Kandy**: 25,899 cases
5. **Kurunegala**: 22,537 cases

</details>

<details>
<summary>⚙️ System Requirements</summary>

- **Python**: 3.8+
- **RAM**: 4GB minimum  
- **Storage**: 1GB
- **Browser**: Any modern browser

</details>

---

## 🤝 Contributing

Found a bug? Have ideas? 

1. 🍴 Fork this repo
2. 🌟 Make it better
3. 📤 Send a pull request

---

## ⚠️ Important Note

This tool is for **research and planning** purposes. For medical decisions, consult healthcare professionals.

---

## 🎯 Quick Links

- 🚀 [Launch Dashboard](#-quick-start)
- 🎮 [How to Use](#-using-the-dashboard)  
- 🧠 [About the AI](#-the-ai-model)
- 💡 [Pro Tips](#-pro-tips)

---

<div align="center">

**Made with ❤️ for dengue prevention in Sri Lanka**

⭐ Star this repo if it helped you!

</div> 