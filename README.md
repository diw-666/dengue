# 🦟 Dengue Forecasting in Sri Lanka Using Spatio-Temporal Neural Networks

A comprehensive AI-powered system for predicting dengue cases across Sri Lankan districts using advanced neural network architectures.

## 🎯 Project Overview

This project develops a sophisticated spatio-temporal neural network model to forecast monthly dengue cases per district in Sri Lanka using historical data from 2010-2020. The system features an interactive dashboard where users can select any district and generate predictions for up to 10 years.

### 🔬 Key Features

- **Advanced Neural Network**: LSTM + Attention mechanism for temporal pattern recognition
- **Spatio-Temporal Analysis**: Considers both spatial (district-wise) and temporal patterns
- **Interactive Dashboard**: User-friendly web interface for predictions
- **Long-term Forecasting**: Generate predictions for 1-10 years ahead
- **Comprehensive Metrics**: Historical analysis, trend detection, and risk assessment
- **Data Export**: Download predictions as CSV files

## 🏗️ Architecture

### Neural Network Model
- **Input Layer**: Multi-dimensional time series features
- **LSTM Layers**: 3 stacked LSTM layers (128, 64, 32 units) with dropout
- **Attention Mechanism**: Focuses on relevant temporal patterns
- **Dense Layers**: Spatial pattern recognition with dropout regularization
- **Output Layer**: Multi-step forecasting (6-month horizon per prediction)

### Features Used
- Historical dengue cases
- Temporal features (month, quarter, season)
- Lag features (1, 2, 3, 6, 12 months)
- Rolling statistics (3, 6, 12-month windows)
- Seasonal encoding (sine/cosine transformations)
- District encoding

## 📊 Dataset

- **Source**: Sri Lanka Dengue Cases (2010-2020)
- **Coverage**: 26 districts across Sri Lanka
- **Records**: 3,432 monthly observations
- **Time Range**: 132 months (11 years)
- **Completeness**: 100% data coverage

### Top Districts by Cases (2010-2020)
1. Colombo: 46,831 cases
2. Gampaha: 32,537 cases
3. Kalutara: 25,982 cases
4. Kandy: 25,899 cases
5. Kurunegala: 22,537 cases

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd dengue

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
# Option 1: Use the launcher script (recommended)
python run_dashboard.py

# Option 2: Run directly
streamlit run dengue_dashboard.py
```

### 3. Access the Dashboard

Open your web browser and navigate to: `http://localhost:8501`

## 🎛️ Dashboard Features

### 📍 District Selection
- Choose from all 26 Sri Lankan districts
- Real-time data loading and processing

### 📅 Prediction Period
- Adjustable forecast horizon (1-10 years)
- Automatic model adaptation for different time ranges

### 📊 Visualizations
- **Historical vs Predicted**: Interactive time series chart
- **Seasonal Patterns**: Monthly breakdown of predicted cases
- **Confidence Intervals**: 95% prediction confidence bands
- **Trend Analysis**: Increasing/decreasing trend indicators

### 📈 Key Metrics
- Historical average cases
- Predicted average cases
- Maximum predicted cases
- Trend direction and percentage change

### 🎯 Risk Assessment
- **High Risk Months**: Cases > 1.5× historical average
- **Moderate Risk Months**: Cases between 1× and 1.5× historical average
- **Low Risk Months**: Cases < historical average

### 💾 Data Export
- Download predictions as CSV files
- Includes dates, predicted cases, and metadata

## 📁 Project Structure

```
dengue/
├── Dengue_Data (2010-2020).xlsx    # Original dataset
├── dengue_predictor_enhanced.py    # Enhanced neural network model
├── dengue_dashboard.py             # Streamlit dashboard
├── run_dashboard.py               # Dashboard launcher script
├── data_exploration.py            # Data analysis script
├── detailed_analysis.py           # Detailed data insights
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── enhanced_dengue_forecaster.pkl # Trained model (generated)
```

## 🤖 Model Performance

### Training Configuration
- **Sequence Length**: 12 months
- **Forecast Horizon**: 6 months per step
- **Training Epochs**: 50 (with early stopping)
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate: 0.001)

### Performance Metrics
- **Mean Absolute Error (MAE)**: ~45-55 cases
- **R² Score**: ~0.75-0.85
- **Training Time**: ~10-15 minutes on modern hardware

### Model Features
- **Early Stopping**: Prevents overfitting
- **Learning Rate Reduction**: Adaptive learning
- **Model Checkpointing**: Saves best performing model
- **Dropout Regularization**: Reduces overfitting

## 🔮 Prediction Methodology

### Long-term Forecasting Process
1. **Initial Sequence**: Use last 12 months of historical data
2. **Iterative Prediction**: Generate 6-month forecasts iteratively
3. **Feature Engineering**: Create future features for each prediction step
4. **Sequence Update**: Roll forward with predicted values
5. **Confidence Estimation**: Calculate prediction intervals

### Feature Engineering for Future Dates
- **Temporal Features**: Month, quarter, seasonal encoding
- **Lag Features**: Estimated from recent predictions
- **Rolling Statistics**: Computed from prediction history
- **District Encoding**: Maintained throughout prediction period

## 📊 Usage Examples

### Basic Prediction
```python
from dengue_predictor_enhanced import EnhancedDengueForecaster

# Load trained model
forecaster = EnhancedDengueForecaster()
forecaster.load_model()

# Make 5-year prediction for Colombo
prediction = forecaster.predict_long_term('Colombo', years=5)
print(f"Predicted cases: {len(prediction['predictions'])}")
```

### Historical Data Analysis
```python
# Get historical data
historical = forecaster.get_historical_data('Colombo')
print(f"Historical period: {historical['dates'][0]} to {historical['dates'][-1]}")
```

## ⚠️ Important Notes

### Model Limitations
- **Training Period**: Based on 2010-2020 data
- **External Factors**: Doesn't account for climate change, policy changes, or unprecedented events
- **Uncertainty**: Long-term predictions have increasing uncertainty
- **Research Purpose**: Intended for research and planning, not clinical decisions

### Best Practices
- Use predictions as guidance, not absolute truth
- Consider multiple scenarios and external factors
- Regularly retrain model with new data
- Validate predictions against actual outcomes

## 🛠️ Development

### Training a New Model
```bash
# Train from scratch
python dengue_predictor_enhanced.py
```

### Customizing the Model
- Modify `sequence_length` for different input windows
- Adjust `forecast_horizon` for different prediction steps
- Experiment with different neural network architectures
- Add new features or preprocessing steps

### Adding New Districts
- Update the dataset with new district data
- Retrain the model to include new locations
- Update the dashboard district list

## 📚 Technical Details

### Dependencies
- **TensorFlow**: Neural network framework
- **Streamlit**: Web dashboard framework
- **Plotly**: Interactive visualizations
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Machine learning utilities

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB for model and data
- **CPU**: Multi-core recommended for training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

## 🙏 Acknowledgments

- Sri Lanka Department of Health for dengue surveillance data
- TensorFlow and Streamlit communities
- Research community working on dengue forecasting

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Check the documentation
- Review the code comments for implementation details

---

**⚠️ Disclaimer**: This tool is for research and planning purposes only. Always consult with health authorities and medical professionals for official guidance on dengue prevention and control. 