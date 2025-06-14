import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import pickle
from datetime import datetime, timedelta
import os
warnings.filterwarnings('ignore')

class EnhancedDengueForecaster:
    def __init__(self, sequence_length=12, forecast_horizon=6):
        """
        Enhanced Dengue Forecaster with long-term prediction capabilities
        
        Args:
            sequence_length: Number of past months to use for prediction
            forecast_horizon: Number of months to forecast ahead per step
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.district_encoder = LabelEncoder()
        self.model = None
        self.districts = None
        self.history = None
        self.df = None
        self.feature_cols = None
        
    def load_and_prepare_data(self, filepath):
        """Load and prepare the dengue data"""
        print("📊 Loading and preparing data...")
        
        # Load data
        df = pd.read_excel(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['City', 'Date']).reset_index(drop=True)
        
        # Add temporal features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Season'] = df['Month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                       3: 'Spring', 4: 'Spring', 5: 'Spring',
                                       6: 'Summer', 7: 'Summer', 8: 'Summer',
                                       9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
        
        # Encode districts
        df['District_Code'] = self.district_encoder.fit_transform(df['City'])
        self.districts = df['City'].unique()
        
        # Create lag features
        for lag in [1, 2, 3, 6, 12]:
            df[f'Cases_Lag_{lag}'] = df.groupby('City')['Value'].shift(lag)
        
        # Create rolling statistics
        for window in [3, 6, 12]:
            df[f'Cases_Rolling_Mean_{window}'] = df.groupby('City')['Value'].rolling(window=window).mean().reset_index(0, drop=True)
            df[f'Cases_Rolling_Std_{window}'] = df.groupby('City')['Value'].rolling(window=window).std().reset_index(0, drop=True)
        
        # Create seasonal features
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        self.df = df.dropna().reset_index(drop=True)
        
        # Define feature columns
        self.feature_cols = ['Value', 'Month', 'Quarter', 'District_Code', 'Month_Sin', 'Month_Cos'] + \
                           [col for col in self.df.columns if 'Lag_' in col or 'Rolling_' in col]
        
        print(f"✅ Data prepared: {len(self.df)} records, {len(self.districts)} districts")
        print(f"📅 Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        return self.df
    
    def create_sequences(self, data, target_col='Value'):
        """Create sequences for time series prediction"""
        print(f"🔄 Creating sequences (length={self.sequence_length})...")
        
        sequences = []
        targets = []
        districts = []
        dates = []
        
        for district in data['City'].unique():
            district_data = data[data['City'] == district].sort_values('Date')
            
            if len(district_data) < self.sequence_length + self.forecast_horizon:
                continue
                
            district_features = district_data[self.feature_cols].values
            district_targets = district_data[target_col].values
            
            for i in range(len(district_data) - self.sequence_length - self.forecast_horizon + 1):
                seq = district_features[i:i + self.sequence_length]
                target = district_targets[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
                
                sequences.append(seq)
                targets.append(target)
                districts.append(district)
                dates.append(district_data.iloc[i + self.sequence_length]['Date'])
        
        return np.array(sequences), np.array(targets), districts, dates
    
    def build_model(self, input_shape):
        """Build the spatio-temporal neural network model"""
        print("🏗️  Building enhanced spatio-temporal neural network...")
        
        # Input layer
        inputs = keras.Input(shape=input_shape)
        
        # LSTM layers for temporal patterns
        lstm1 = layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        lstm2 = layers.LSTM(64, return_sequences=True, dropout=0.2)(lstm1)
        lstm3 = layers.LSTM(32, return_sequences=False, dropout=0.2)(lstm2)
        
        # Attention mechanism
        attention = layers.Dense(32, activation='tanh')(lstm3)
        attention = layers.Dense(1, activation='softmax')(attention)
        context = layers.multiply([lstm3, attention])
        
        # Dense layers for spatial patterns
        dense1 = layers.Dense(64, activation='relu')(context)
        dense1 = layers.Dropout(0.3)(dense1)
        dense2 = layers.Dense(32, activation='relu')(dense1)
        dense2 = layers.Dropout(0.2)(dense2)
        
        # Output layer for multi-step forecasting
        outputs = layers.Dense(self.forecast_horizon, activation='linear')(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with custom metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the neural network model"""
        print("🚀 Training the enhanced model...")
        
        # Build model
        self.model = self.build_model(X_train.shape[1:])
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint('enhanced_dengue_model.h5', save_best_only=True)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Enhanced model training completed!")
        return self.history
    
    def predict_long_term(self, district, years=5):
        """
        Make long-term predictions for a specific district
        
        Args:
            district: Name of the district
            years: Number of years to predict ahead
            
        Returns:
            Dictionary with dates and predictions
        """
        print(f"🔮 Making {years}-year predictions for {district}...")
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get the latest data for the district
        district_data = self.df[self.df['City'] == district].sort_values('Date')
        
        if len(district_data) < self.sequence_length:
            raise ValueError(f"Not enough data for {district}")
        
        # Get the last sequence
        last_sequence = district_data[self.feature_cols].values[-self.sequence_length:]
        
        # Generate predictions
        predictions = []
        current_sequence = last_sequence.copy()
        
        # Start date for predictions
        last_date = district_data['Date'].iloc[-1]
        prediction_dates = []
        
        months_to_predict = years * 12
        steps = months_to_predict // self.forecast_horizon
        remaining_months = months_to_predict % self.forecast_horizon
        
        for step in range(steps):
            # Make prediction
            pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, -1), verbose=0)
            predictions.extend(pred[0])
            
            # Update sequence for next prediction
            for i in range(self.forecast_horizon):
                # Create next month's features
                next_date = last_date + timedelta(days=32 * (step * self.forecast_horizon + i + 1))
                next_date = next_date.replace(day=1)  # First day of month
                
                prediction_dates.append(next_date)
                
                # Create feature vector for next month
                next_features = self._create_future_features(
                    district, next_date, pred[0][i], current_sequence
                )
                
                # Update sequence (shift and add new features)
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = next_features
        
        # Handle remaining months if any
        if remaining_months > 0:
            pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, -1), verbose=0)
            predictions.extend(pred[0][:remaining_months])
            
            for i in range(remaining_months):
                next_date = last_date + timedelta(days=32 * (steps * self.forecast_horizon + i + 1))
                next_date = next_date.replace(day=1)
                prediction_dates.append(next_date)
        
        return {
            'dates': prediction_dates,
            'predictions': predictions,
            'district': district
        }
    
    def _create_future_features(self, district, date, predicted_value, current_sequence):
        """Create feature vector for future date"""
        # Basic temporal features
        month = date.month
        quarter = (month - 1) // 3 + 1
        district_code = self.district_encoder.transform([district])[0]
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Create lag features (use recent predictions and historical data)
        recent_values = current_sequence[-12:, 0]  # Last 12 months of cases
        
        # Estimate lag features
        lag_1 = recent_values[-1] if len(recent_values) >= 1 else predicted_value
        lag_2 = recent_values[-2] if len(recent_values) >= 2 else predicted_value
        lag_3 = recent_values[-3] if len(recent_values) >= 3 else predicted_value
        lag_6 = recent_values[-6] if len(recent_values) >= 6 else predicted_value
        lag_12 = recent_values[0] if len(recent_values) >= 12 else predicted_value
        
        # Rolling statistics
        rolling_mean_3 = np.mean(recent_values[-3:]) if len(recent_values) >= 3 else predicted_value
        rolling_mean_6 = np.mean(recent_values[-6:]) if len(recent_values) >= 6 else predicted_value
        rolling_mean_12 = np.mean(recent_values) if len(recent_values) >= 12 else predicted_value
        
        rolling_std_3 = np.std(recent_values[-3:]) if len(recent_values) >= 3 else 0
        rolling_std_6 = np.std(recent_values[-6:]) if len(recent_values) >= 6 else 0
        rolling_std_12 = np.std(recent_values) if len(recent_values) >= 12 else 0
        
        # Combine features in the same order as training data
        features = [
            predicted_value,  # Value
            month,           # Month
            quarter,         # Quarter
            district_code,   # District_Code
            month_sin,       # Month_Sin
            month_cos,       # Month_Cos
            lag_1,           # Cases_Lag_1
            lag_2,           # Cases_Lag_2
            lag_3,           # Cases_Lag_3
            lag_6,           # Cases_Lag_6
            lag_12,          # Cases_Lag_12
            rolling_mean_3,  # Cases_Rolling_Mean_3
            rolling_mean_6,  # Cases_Rolling_Mean_6
            rolling_mean_12, # Cases_Rolling_Mean_12
            rolling_std_3,   # Cases_Rolling_Std_3
            rolling_std_6,   # Cases_Rolling_Std_6
            rolling_std_12   # Cases_Rolling_Std_12
        ]
        
        return np.array(features)
    
    def get_historical_data(self, district):
        """Get historical data for a district"""
        district_data = self.df[self.df['City'] == district].sort_values('Date')
        return {
            'dates': district_data['Date'].tolist(),
            'values': district_data['Value'].tolist(),
            'district': district
        }
    
    def save_model(self, filepath='enhanced_dengue_forecaster.pkl'):
        """Save the complete forecaster"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'district_encoder': self.district_encoder,
            'districts': self.districts,
            'feature_cols': self.feature_cols,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'df': self.df
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='enhanced_dengue_forecaster.pkl'):
        """Load the complete forecaster"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.district_encoder = model_data['district_encoder']
        self.districts = model_data['districts']
        self.feature_cols = model_data['feature_cols']
        self.sequence_length = model_data['sequence_length']
        self.forecast_horizon = model_data['forecast_horizon']
        self.df = model_data['df']
        
        print(f"✅ Model loaded from {filepath}")

def train_enhanced_model():
    """Train the enhanced model if not already trained"""
    print("🦟 ENHANCED DENGUE FORECASTING IN SRI LANKA")
    print("=" * 60)
    
    # Initialize forecaster
    forecaster = EnhancedDengueForecaster(sequence_length=12, forecast_horizon=6)
    
    # Load and prepare data
    data = forecaster.load_and_prepare_data('Dengue_Data (2010-2020).xlsx')
    
    # Create sequences
    X, y, districts, dates = forecaster.create_sequences(data)
    
    print(f"📊 Dataset shape: X={X.shape}, y={y.shape}")
    
    # Split data (temporal split - last 20% for testing)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"🔄 Data splits:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples") 
    print(f"  Testing: {X_test.shape[0]} samples")
    
    # Train model
    history = forecaster.train_model(X_train, y_train, X_val, y_val, epochs=50)
    
    # Evaluate model
    print("📊 Evaluating model performance...")
    y_pred = forecaster.model.predict(X_test)
    
    # Calculate overall metrics
    overall_mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    overall_r2 = r2_score(y_test.flatten(), y_pred.flatten())
    
    print(f"📈 Overall Performance:")
    print(f"  MAE: {overall_mae:.4f}")
    print(f"  R²: {overall_r2:.4f}")
    
    # Save the complete model
    forecaster.save_model()
    
    print("🎉 Enhanced dengue forecasting model completed successfully!")
    return forecaster

if __name__ == "__main__":
    # Check if model exists, if not train it
    if not os.path.exists('enhanced_dengue_forecaster.pkl'):
        forecaster = train_enhanced_model()
    else:
        print("✅ Using existing trained model")
        forecaster = EnhancedDengueForecaster()
        forecaster.load_model()
    
    # Example: Make 5-year prediction for Colombo
    try:
        prediction = forecaster.predict_long_term('Colombo', years=5)
        print(f"\n🔮 Sample 5-year prediction for Colombo:")
        print(f"  Prediction period: {prediction['dates'][0]} to {prediction['dates'][-1]}")
        print(f"  Average predicted cases: {np.mean(prediction['predictions']):.1f}")
        print(f"  Max predicted cases: {np.max(prediction['predictions']):.1f}")
        print(f"  Min predicted cases: {np.min(prediction['predictions']):.1f}")
    except Exception as e:
        print(f"❌ Error making prediction: {e}") 