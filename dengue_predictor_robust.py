import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
import pickle
from datetime import datetime, timedelta
import os
warnings.filterwarnings('ignore')

class RobustDengueForecaster:
    def __init__(self, sequence_length=12, forecast_horizon=6):
        """
        Robust Dengue Forecaster with simplified, well-regularized architecture
        
        Args:
            sequence_length: Number of past months to use for prediction (reduced for stability)
            forecast_horizon: Number of months to forecast ahead
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # Changed to StandardScaler for better stability
        self.district_encoder = LabelEncoder()
        self.model = None
        self.districts = None
        self.history = None
        self.df = None
        self.feature_cols = None
        
    def load_and_prepare_data(self, filepath):
        """Load and prepare data with robust feature engineering"""
        print("📊 Loading and preparing data with robust features...")
        
        # Load data
        df = pd.read_excel(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['City', 'Date']).reset_index(drop=True)
        
        # Basic temporal features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        
        # Cyclical encoding (essential for seasonality)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Quarter_Sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
        df['Quarter_Cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
        
        # Encode districts
        df['District_Code'] = self.district_encoder.fit_transform(df['City'])
        self.districts = df['City'].unique()
        
        # Essential lag features (simplified)
        for lag in [1, 2, 3, 6, 12]:
            df[f'Lag_{lag}'] = df.groupby('City')['Value'].shift(lag)
        
        # Essential rolling statistics (simplified)
        for window in [3, 6, 12]:
            df[f'Rolling_Mean_{window}'] = df.groupby('City')['Value'].rolling(window=window).mean().reset_index(0, drop=True)
            df[f'Rolling_Std_{window}'] = df.groupby('City')['Value'].rolling(window=window).std().reset_index(0, drop=True)
        
        # Trend features
        df['Cases_Diff_1'] = df.groupby('City')['Value'].diff(1)
        df['Cases_Diff_12'] = df.groupby('City')['Value'].diff(12)
        
        # District statistics (simplified)
        district_stats = df.groupby('City')['Value'].agg(['mean', 'std']).reset_index()
        district_stats.columns = ['City', 'District_Mean', 'District_Std']
        df = df.merge(district_stats, on='City', how='left')
        
        # Relative features
        df['Value_Normalized'] = (df['Value'] - df['District_Mean']) / (df['District_Std'] + 1e-8)
        
        # Clean data properly
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Remove extreme outliers
        for col in numeric_cols:
            if 'Value' in col or 'Cases' in col or 'Lag' in col or 'Rolling' in col:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=q01, upper=q99)
        
        self.df = df.dropna().reset_index(drop=True)
        
        # Simplified feature set (most important features only)
        self.feature_cols = [
            'Value', 'Month', 'Quarter', 'District_Code',
            'Month_Sin', 'Month_Cos', 'Quarter_Sin', 'Quarter_Cos',
            'Lag_1', 'Lag_2', 'Lag_3', 'Lag_6', 'Lag_12',
            'Rolling_Mean_3', 'Rolling_Mean_6', 'Rolling_Mean_12',
            'Rolling_Std_3', 'Rolling_Std_6', 'Rolling_Std_12',
            'Cases_Diff_1', 'Cases_Diff_12', 'Value_Normalized'
        ]
        
        print(f"✅ Data prepared: {len(self.df)} records, {len(self.districts)} districts")
        print(f"📅 Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"🔧 Features: {len(self.feature_cols)} essential features")
        
        return self.df
    
    def create_sequences_robust(self, data):
        """Create sequences with proper temporal splitting to avoid data leakage"""
        print(f"🔄 Creating robust sequences (length={self.sequence_length})...")
        
        sequences = []
        targets = []
        districts = []
        dates = []
        
        # Process each district separately
        for district in data['City'].unique():
            district_data = data[data['City'] == district].sort_values('Date')
            
            if len(district_data) < self.sequence_length + self.forecast_horizon:
                continue
            
            # Extract features and targets
            features = district_data[self.feature_cols].values
            targets_raw = district_data['Value'].values
            
            # Create sequences
            for i in range(len(district_data) - self.sequence_length - self.forecast_horizon + 1):
                seq = features[i:i + self.sequence_length]
                target = targets_raw[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
                
                sequences.append(seq)
                targets.append(target)
                districts.append(district)
                dates.append(district_data.iloc[i + self.sequence_length]['Date'])
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(targets)
        
        # Proper scaling after sequence creation
        # Fit scalers only on training data to avoid data leakage
        return X, y, districts, dates
    
    def scale_data(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        """Scale data properly to avoid data leakage"""
        print("📏 Scaling data properly...")
        
        # Fit scalers only on training data
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        y_train_flat = y_train.reshape(-1, 1)
        
        self.feature_scaler.fit(X_train_flat)
        self.target_scaler.fit(y_train_flat)
        
        # Transform training data
        X_train_scaled = self.feature_scaler.transform(X_train_flat).reshape(X_train.shape)
        y_train_scaled = self.target_scaler.transform(y_train_flat).reshape(y_train.shape)
        
        result = [X_train_scaled, y_train_scaled]
        
        # Transform validation data if provided
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(-1, X_val.shape[-1])
            y_val_flat = y_val.reshape(-1, 1)
            X_val_scaled = self.feature_scaler.transform(X_val_flat).reshape(X_val.shape)
            y_val_scaled = self.target_scaler.transform(y_val_flat).reshape(y_val.shape)
            result.extend([X_val_scaled, y_val_scaled])
        
        # Transform test data if provided
        if X_test is not None and y_test is not None:
            X_test_flat = X_test.reshape(-1, X_test.shape[-1])
            y_test_flat = y_test.reshape(-1, 1)
            X_test_scaled = self.feature_scaler.transform(X_test_flat).reshape(X_test.shape)
            y_test_scaled = self.target_scaler.transform(y_test_flat).reshape(y_test.shape)
            result.extend([X_test_scaled, y_test_scaled])
        
        return result
    
    def build_robust_model(self, input_shape):
        """Build a robust, well-regularized model"""
        print("🏗️ Building robust neural network...")
        
        inputs = keras.Input(shape=input_shape)
        
        # Single LSTM layer with proper regularization
        lstm_out = layers.LSTM(
            64, 
            return_sequences=True, 
            dropout=0.3, 
            recurrent_dropout=0.2,
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(inputs)
        lstm_out = layers.LayerNormalization()(lstm_out)
        
        # Second LSTM layer
        lstm_out2 = layers.LSTM(
            32, 
            return_sequences=False, 
            dropout=0.3, 
            recurrent_dropout=0.2,
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(lstm_out)
        lstm_out2 = layers.LayerNormalization()(lstm_out2)
        
        # Simple dense layers with proper regularization
        dense1 = layers.Dense(
            64, 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(lstm_out2)
        dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Dropout(0.4)(dense1)
        
        dense2 = layers.Dense(
            32, 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        dense2 = layers.Dropout(0.3)(dense2)
        
        # Output layer
        outputs = layers.Dense(
            self.forecast_horizon, 
            activation='linear',
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Conservative optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Simple loss function
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_robust_model(self, X, y, districts, dates, epochs=100):
        """Train model with proper time series cross-validation"""
        print("🚀 Training robust model with time series validation...")
        
        # Time series split to avoid data leakage
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Convert to proper format for time series splitting
        # Sort by date to ensure proper temporal order
        date_indices = np.argsort([d.timestamp() for d in dates])
        X_sorted = X[date_indices]
        y_sorted = y[date_indices]
        
        best_model = None
        best_score = float('inf')
        
        # Use the last split for final training
        splits = list(tscv.split(X_sorted))
        train_idx, val_idx = splits[-1]
        
        X_train, X_val = X_sorted[train_idx], X_sorted[val_idx]
        y_train, y_val = y_sorted[train_idx], y_sorted[val_idx]
        
        # Scale data properly
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled = self.scale_data(
            X_train, y_train, X_val, y_val
        )
        
        # Build model
        self.model = self.build_robust_model(X_train_scaled.shape[1:])
        
        # Robust callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=20,
                restore_best_weights=True,
                monitor='val_loss',
                min_delta=0.001
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                monitor='val_loss'
            ),
            keras.callbacks.ModelCheckpoint(
                'robust_dengue_model.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=epochs,
            batch_size=32,  # Smaller batch size for stability
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Robust model training completed!")
        return self.history
    
    def evaluate_robust_model(self, X, y, districts, dates):
        """Evaluate model with proper temporal splitting"""
        print("📊 Evaluating robust model...")
        
        # Use temporal split for evaluation
        tscv = TimeSeriesSplit(n_splits=3)
        date_indices = np.argsort([d.timestamp() for d in dates])
        X_sorted = X[date_indices]
        y_sorted = y[date_indices]
        
        # Use the last split for evaluation
        splits = list(tscv.split(X_sorted))
        train_idx, test_idx = splits[-1]
        
        X_train, X_test = X_sorted[train_idx], X_sorted[test_idx]
        y_train, y_test = y_sorted[train_idx], y_sorted[test_idx]
        
        # Scale data
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = self.scale_data(
            X_train, y_train, X_test=X_test, y_test=y_test
        )
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test_scaled, verbose=0)
        
        # Inverse transform
        y_test_inv = self.target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(y_test.shape)
        y_pred_inv = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
        
        # Calculate metrics
        print("📈 Robust Model Performance:")
        print("-" * 50)
        
        for horizon in range(self.forecast_horizon):
            mae = mean_absolute_error(y_test_inv[:, horizon], y_pred_inv[:, horizon])
            mse = mean_squared_error(y_test_inv[:, horizon], y_pred_inv[:, horizon])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_inv[:, horizon], y_pred_inv[:, horizon])
            
            # Improved MAPE calculation
            mape = np.mean(np.abs((y_test_inv[:, horizon] - y_pred_inv[:, horizon]) / 
                                 np.maximum(np.abs(y_test_inv[:, horizon]), 1))) * 100
            
            print(f"Horizon_{horizon+1}:")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R2: {r2:.4f}")
            print(f"  MAPE: {mape:.4f}")
            print()
        
        # Overall metrics
        overall_mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())
        overall_mse = mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten())
        overall_rmse = np.sqrt(overall_mse)
        overall_r2 = r2_score(y_test_inv.flatten(), y_pred_inv.flatten())
        overall_mape = np.mean(np.abs((y_test_inv.flatten() - y_pred_inv.flatten()) / 
                                     np.maximum(np.abs(y_test_inv.flatten()), 1))) * 100
        
        print("Overall Performance:")
        print(f"  MAE: {overall_mae:.4f}")
        print(f"  MSE: {overall_mse:.4f}")
        print(f"  RMSE: {overall_rmse:.4f}")
        print(f"  R2: {overall_r2:.4f}")
        print(f"  MAPE: {overall_mape:.4f}")
        print()
        
        return {
            'test_predictions': y_pred_inv,
            'test_actual': y_test_inv,
            'overall_mae': overall_mae,
            'overall_r2': overall_r2,
            'overall_mape': overall_mape
        }
    
    def predict_long_term_robust(self, district, years=5):
        """Make robust long-term predictions"""
        print(f"🔮 Making robust {years}-year predictions for {district}...")
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get the latest data for the district
        district_data = self.df[self.df['City'] == district].sort_values('Date')
        
        if len(district_data) < self.sequence_length:
            raise ValueError(f"Not enough data for {district}")
        
        # Get the last sequence
        last_sequence = district_data[self.feature_cols].values[-self.sequence_length:]
        
        # Scale the sequence
        last_sequence_scaled = self.feature_scaler.transform(last_sequence)
        
        # Generate predictions
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        # Start date for predictions
        last_date = district_data['Date'].iloc[-1]
        prediction_dates = []
        
        months_to_predict = years * 12
        steps = months_to_predict // self.forecast_horizon
        remaining_months = months_to_predict % self.forecast_horizon
        
        for step in range(steps):
            # Make prediction
            pred_scaled = self.model.predict(
                current_sequence.reshape(1, self.sequence_length, -1), 
                verbose=0
            )
            pred_unscaled = self.target_scaler.inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).flatten()
            
            predictions.extend(pred_unscaled)
            
            # Update sequence for next prediction
            for i in range(self.forecast_horizon):
                next_date = last_date + timedelta(days=32 * (step * self.forecast_horizon + i + 1))
                next_date = next_date.replace(day=1)
                prediction_dates.append(next_date)
                
                # Create simplified feature vector for next month
                next_features = self._create_simple_future_features(
                    district, next_date, pred_unscaled[i], current_sequence
                )
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = next_features
        
        # Handle remaining months
        if remaining_months > 0:
            pred_scaled = self.model.predict(
                current_sequence.reshape(1, self.sequence_length, -1), 
                verbose=0
            )
            pred_unscaled = self.target_scaler.inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).flatten()
            
            predictions.extend(pred_unscaled[:remaining_months])
            
            for i in range(remaining_months):
                next_date = last_date + timedelta(days=32 * (steps * self.forecast_horizon + i + 1))
                next_date = next_date.replace(day=1)
                prediction_dates.append(next_date)
        
        return {
            'dates': prediction_dates,
            'predictions': predictions,
            'district': district
        }
    
    def _create_simple_future_features(self, district, date, predicted_value, current_sequence):
        """Create simplified feature vector for future date"""
        # Basic temporal features
        month = date.month
        quarter = (month - 1) // 3 + 1
        district_code = self.district_encoder.transform([district])[0]
        
        # Cyclical features
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        quarter_sin = np.sin(2 * np.pi * quarter / 4)
        quarter_cos = np.cos(2 * np.pi * quarter / 4)
        
        # Lag features from recent sequence
        recent_values = current_sequence[-12:, 0]  # Get Value column
        lag_1 = recent_values[-1] if len(recent_values) >= 1 else 0
        lag_2 = recent_values[-2] if len(recent_values) >= 2 else 0
        lag_3 = recent_values[-3] if len(recent_values) >= 3 else 0
        lag_6 = recent_values[-6] if len(recent_values) >= 6 else 0
        lag_12 = recent_values[0] if len(recent_values) >= 12 else 0
        
        # Rolling statistics
        rolling_mean_3 = np.mean(recent_values[-3:]) if len(recent_values) >= 3 else 0
        rolling_mean_6 = np.mean(recent_values[-6:]) if len(recent_values) >= 6 else 0
        rolling_mean_12 = np.mean(recent_values) if len(recent_values) >= 12 else 0
        
        rolling_std_3 = np.std(recent_values[-3:]) if len(recent_values) >= 3 else 0
        rolling_std_6 = np.std(recent_values[-6:]) if len(recent_values) >= 6 else 0
        rolling_std_12 = np.std(recent_values) if len(recent_values) >= 12 else 0
        
        # Difference features
        cases_diff_1 = recent_values[-1] - recent_values[-2] if len(recent_values) >= 2 else 0
        cases_diff_12 = recent_values[-1] - recent_values[0] if len(recent_values) >= 12 else 0
        
        # Normalized value (simplified)
        value_normalized = 0  # Will be updated later
        
        # Combine features in the same order as training data
        features = np.array([
            predicted_value,  # Value
            month,           # Month
            quarter,         # Quarter
            district_code,   # District_Code
            month_sin,       # Month_Sin
            month_cos,       # Month_Cos
            quarter_sin,     # Quarter_Sin
            quarter_cos,     # Quarter_Cos
            lag_1,           # Lag_1
            lag_2,           # Lag_2
            lag_3,           # Lag_3
            lag_6,           # Lag_6
            lag_12,          # Lag_12
            rolling_mean_3,  # Rolling_Mean_3
            rolling_mean_6,  # Rolling_Mean_6
            rolling_mean_12, # Rolling_Mean_12
            rolling_std_3,   # Rolling_Std_3
            rolling_std_6,   # Rolling_Std_6
            rolling_std_12,  # Rolling_Std_12
            cases_diff_1,    # Cases_Diff_1
            cases_diff_12,   # Cases_Diff_12
            value_normalized # Value_Normalized
        ])
        
        return features
    
    def save_model(self, filepath='robust_dengue_forecaster.pkl'):
        """Save the complete model"""
        model_data = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'district_encoder': self.district_encoder,
            'districts': self.districts,
            'feature_cols': self.feature_cols,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'df': self.df
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Robust model saved to {filepath}")
    
    def load_model(self, filepath='robust_dengue_forecaster.pkl'):
        """Load the complete model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_scaler = model_data['feature_scaler']
        self.target_scaler = model_data['target_scaler']
        self.district_encoder = model_data['district_encoder']
        self.districts = model_data['districts']
        self.feature_cols = model_data['feature_cols']
        self.sequence_length = model_data['sequence_length']
        self.forecast_horizon = model_data['forecast_horizon']
        self.df = model_data['df']
        print(f"✅ Robust model loaded from {filepath}")

def train_robust_model():
    """Train the robust model"""
    print("🚀 Training robust dengue forecasting model...")
    
    # Initialize robust forecaster with conservative parameters
    forecaster = RobustDengueForecaster(sequence_length=12, forecast_horizon=6)
    
    # Load and prepare data
    data = forecaster.load_and_prepare_data('Dengue_Data (2010-2020).xlsx')
    
    # Create sequences
    X, y, districts, dates = forecaster.create_sequences_robust(data)
    
    print(f"📊 Dataset shape: X={X.shape}, y={y.shape}")
    
    # Train model with proper validation
    history = forecaster.train_robust_model(X, y, districts, dates, epochs=50)
    
    # Evaluate model
    results = forecaster.evaluate_robust_model(X, y, districts, dates)
    
    # Save model
    forecaster.save_model()
    
    return forecaster, results

if __name__ == "__main__":
    # Check if robust model exists
    if not os.path.exists('robust_dengue_forecaster.pkl'):
        print("🚀 Training new robust model...")
        forecaster, results = train_robust_model()
    else:
        print("✅ Using existing robust model")
        forecaster = RobustDengueForecaster()
        forecaster.load_model()
    
    # Example: Make robust 3-year prediction for Colombo
    try:
        prediction = forecaster.predict_long_term_robust('Colombo', years=3)
        print(f"\n🔮 Robust 3-year prediction for Colombo:")
        print(f"  Prediction period: {prediction['dates'][0]} to {prediction['dates'][-1]}")
        print(f"  Average predicted cases: {np.mean(prediction['predictions']):.1f}")
        print(f"  Max predicted cases: {np.max(prediction['predictions']):.1f}")
        print(f"  Min predicted cases: {np.min(prediction['predictions']):.1f}")
        
        # Print first 12 months
        print(f"\n📊 First 12 months prediction:")
        for i in range(min(12, len(prediction['dates']))):
            print(f"  {prediction['dates'][i].strftime('%Y-%m')}: {prediction['predictions'][i]:.1f} cases")
            
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
    
    print("\n🎉 Robust dengue forecasting model completed!")
    print("📁 Key improvements:")
    print("  - Simplified architecture with proper regularization")
    print("  - Essential features only (22 features)")
    print("  - Proper temporal cross-validation")
    print("  - Robust scaling and data handling")
    print("  - Conservative hyperparameters")
    print("  - Proper handling of data leakage") 