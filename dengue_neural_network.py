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
warnings.filterwarnings('ignore')

print("🦟 DENGUE FORECASTING IN SRI LANKA USING SPATIO-TEMPORAL NEURAL NETWORKS")
print("=" * 80)

class DengueForecaster:
    def __init__(self, sequence_length=12, forecast_horizon=6):
        """
        Initialize the Dengue Forecaster
        
        Args:
            sequence_length: Number of past months to use for prediction
            forecast_horizon: Number of months to forecast ahead
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.district_encoder = LabelEncoder()
        self.model = None
        self.districts = None
        self.history = None
        
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
        
        feature_cols = ['Value', 'Month', 'Quarter', 'District_Code', 'Month_Sin', 'Month_Cos'] + \
                      [col for col in data.columns if 'Lag_' in col or 'Rolling_' in col]
        
        for district in data['City'].unique():
            district_data = data[data['City'] == district].sort_values('Date')
            
            if len(district_data) < self.sequence_length + self.forecast_horizon:
                continue
                
            district_features = district_data[feature_cols].values
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
        print("🏗️  Building spatio-temporal neural network...")
        
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
        
        print("✅ Model architecture:")
        model.summary()
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the neural network model"""
        print("🚀 Training the model...")
        
        # Build model
        self.model = self.build_model(X_train.shape[1:])
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint('best_dengue_model.h5', save_best_only=True)
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
        
        print("✅ Model training completed!")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        print("📊 Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics for each forecast horizon
        metrics = {}
        for horizon in range(self.forecast_horizon):
            mae = mean_absolute_error(y_test[:, horizon], y_pred[:, horizon])
            mse = mean_squared_error(y_test[:, horizon], y_pred[:, horizon])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test[:, horizon], y_pred[:, horizon])
            
            metrics[f'Horizon_{horizon+1}'] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
        
        # Overall metrics
        overall_mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
        overall_mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
        overall_rmse = np.sqrt(overall_mse)
        overall_r2 = r2_score(y_test.flatten(), y_pred.flatten())
        
        metrics['Overall'] = {
            'MAE': overall_mae,
            'MSE': overall_mse,
            'RMSE': overall_rmse,
            'R2': overall_r2
        }
        
        print("📈 Model Performance Metrics:")
        print("-" * 50)
        for horizon, metric_dict in metrics.items():
            print(f"{horizon}:")
            for metric, value in metric_dict.items():
                print(f"  {metric}: {value:.4f}")
            print()
        
        return metrics, y_pred
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("❌ No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        
        # MAPE
        axes[1, 0].plot(self.history.history['mape'], label='Training MAPE')
        axes[1, 0].plot(self.history.history['val_mape'], label='Validation MAPE')
        axes[1, 0].set_title('Mean Absolute Percentage Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAPE')
        axes[1, 0].legend()
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, X_test, y_test, y_pred, districts_test, dates_test, n_samples=5):
        """Plot prediction results"""
        print("📊 Creating prediction visualizations...")
        
        # Select random samples to plot
        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(15, 4*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices):
            district = districts_test[idx]
            date = dates_test[idx]
            
            # Historical data (last part of input sequence)
            historical = X_test[idx, -6:, 0]  # Last 6 months of cases
            
            # Actual vs Predicted
            actual = y_test[idx]
            predicted = y_pred[idx]
            
            # Create time axis
            hist_months = range(-6, 0)
            pred_months = range(0, self.forecast_horizon)
            
            axes[i].plot(hist_months, historical, 'b-', label='Historical', linewidth=2)
            axes[i].plot(pred_months, actual, 'g-', label='Actual', linewidth=2, marker='o')
            axes[i].plot(pred_months, predicted, 'r--', label='Predicted', linewidth=2, marker='s')
            
            axes[i].set_title(f'{district} - Forecast from {date.strftime("%Y-%m")}')
            axes[i].set_xlabel('Months')
            axes[i].set_ylabel('Dengue Cases')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].axvline(x=0, color='black', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('predictions_sample.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, X_test, y_test, y_pred, districts_test, dates_test):
        """Create an interactive dashboard"""
        print("🎯 Creating interactive dashboard...")
        
        # Create district-wise performance summary
        district_performance = {}
        for i, district in enumerate(districts_test):
            if district not in district_performance:
                district_performance[district] = {'actual': [], 'predicted': []}
            district_performance[district]['actual'].extend(y_test[i])
            district_performance[district]['predicted'].extend(y_pred[i])
        
        # Calculate district-wise metrics
        district_metrics = {}
        for district, data in district_performance.items():
            mae = mean_absolute_error(data['actual'], data['predicted'])
            r2 = r2_score(data['actual'], data['predicted'])
            district_metrics[district] = {'MAE': mae, 'R2': r2}
        
        # Create interactive plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('District Performance (MAE)', 'District Performance (R²)', 
                          'Actual vs Predicted', 'Residuals'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # District MAE
        districts_list = list(district_metrics.keys())
        mae_values = [district_metrics[d]['MAE'] for d in districts_list]
        fig.add_trace(
            go.Bar(x=districts_list, y=mae_values, name='MAE'),
            row=1, col=1
        )
        
        # District R²
        r2_values = [district_metrics[d]['R2'] for d in districts_list]
        fig.add_trace(
            go.Bar(x=districts_list, y=r2_values, name='R²'),
            row=1, col=2
        )
        
        # Actual vs Predicted scatter
        all_actual = y_test.flatten()
        all_predicted = y_pred.flatten()
        fig.add_trace(
            go.Scatter(x=all_actual, y=all_predicted, mode='markers', 
                      name='Predictions', opacity=0.6),
            row=2, col=1
        )
        # Perfect prediction line
        min_val, max_val = min(all_actual), max(all_actual)
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction', 
                      line=dict(dash='dash', color='red')),
            row=2, col=1
        )
        
        # Residuals
        residuals = all_actual - all_predicted
        fig.add_trace(
            go.Scatter(x=all_predicted, y=residuals, mode='markers', 
                      name='Residuals', opacity=0.6),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Dengue Forecasting Model Performance Dashboard")
        fig.write_html('dengue_dashboard.html')
        fig.show()
        
        return district_metrics

# Main execution
if __name__ == "__main__":
    # Initialize forecaster
    forecaster = DengueForecaster(sequence_length=12, forecast_horizon=6)
    
    # Load and prepare data
    data = forecaster.load_and_prepare_data('Dengue_Data (2010-2020).xlsx')
    
    # Create sequences
    X, y, districts, dates = forecaster.create_sequences(data)
    
    print(f"📊 Dataset shape: X={X.shape}, y={y.shape}")
    
    # Split data (temporal split - last 20% for testing)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    districts_test = districts[split_idx:]
    dates_test = dates[split_idx:]
    
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
    metrics, y_pred = forecaster.evaluate_model(X_test, y_test)
    
    # Create visualizations
    forecaster.plot_training_history()
    forecaster.plot_predictions(X_test, y_test, y_pred, districts_test, dates_test)
    district_metrics = forecaster.create_interactive_dashboard(X_test, y_test, y_pred, districts_test, dates_test)
    
    print("🎉 Dengue forecasting model completed successfully!")
    print("📁 Generated files:")
    print("  - best_dengue_model.h5 (trained model)")
    print("  - training_history.png (training plots)")
    print("  - predictions_sample.png (sample predictions)")
    print("  - dengue_dashboard.html (interactive dashboard)") 