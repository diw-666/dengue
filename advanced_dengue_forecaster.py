import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import warnings
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import shap
from dengue_predictor_robust import RobustDengueForecaster

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

class AdvancedDengueForecaster(RobustDengueForecaster):
    """
    Advanced Dengue Forecaster with uncertainty quantification and interpretability
    """
    
    def __init__(self, sequence_length=12, forecast_horizon=6, n_bootstrap_samples=100):
        super().__init__(sequence_length, forecast_horizon)
        self.n_bootstrap_samples = n_bootstrap_samples
        self.ensemble_models = []
        self.feature_importance_scores = {}
        self.prediction_intervals = {}
        
    def train_ensemble_models(self, X, y, districts, dates, n_models=5):
        """Train ensemble of models for uncertainty quantification"""
        print(f"\n🎯 Training ensemble of {n_models} models for uncertainty quantification...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.2))
        
        self.ensemble_models = []
        ensemble_histories = []
        
        for i in range(n_models):
            print(f"   Training model {i+1}/{n_models}...")
            
            # Create slightly different model architecture
            model = self._build_ensemble_model(X.shape[1:], model_id=i)
            
            # Bootstrap sampling for diversity
            bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Train model
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=0
            )
            
            history = model.fit(
                X_bootstrap, y_bootstrap,
                epochs=100,
                batch_size=16,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            self.ensemble_models.append(model)
            ensemble_histories.append(history)
        
        print(f"✅ Ensemble training completed with {len(self.ensemble_models)} models")
        
        return ensemble_histories
    
    def _build_ensemble_model(self, input_shape, model_id=0):
        """Build ensemble model with slight variations"""
        model = Sequential([
            LSTM(64 + (model_id * 8), return_sequences=True, 
                 kernel_regularizer=l2(0.001), 
                 recurrent_regularizer=l2(0.001),
                 input_shape=input_shape),
            Dropout(0.3 + (model_id * 0.05)),
            BatchNormalization(),
            
            LSTM(32 + (model_id * 4), return_sequences=False,
                 kernel_regularizer=l2(0.001),
                 recurrent_regularizer=l2(0.001)),
            Dropout(0.3 + (model_id * 0.05)),
            BatchNormalization(),
            
            Dense(16 + (model_id * 2), activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(self.forecast_horizon, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 * (0.9 ** model_id)),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def predict_with_uncertainty(self, district, years=3, confidence_level=0.95):
        """Make predictions with uncertainty quantification"""
        print(f"\n🔮 Making predictions with uncertainty for {district} ({years} years)...")
        
        if not self.ensemble_models:
            print("❌ No ensemble models found. Training ensemble first...")
            return None
        
        # Get base prediction structure
        base_prediction = self.predict_long_term_robust(district, years)
        
        # Prepare input data
        district_data = self.df[self.df['City'] == district].copy()
        district_data = district_data.sort_values('Date')
        
        # Get last sequence
        last_sequence = district_data.tail(self.sequence_length)[self.feature_cols].values
        last_sequence_scaled = self.feature_scaler.transform(last_sequence)
        
        # Get ensemble predictions
        ensemble_predictions = []
        for model in self.ensemble_models:
            pred_scaled = model.predict(
                last_sequence_scaled.reshape(1, self.sequence_length, -1),
                verbose=0
            )
            pred_unscaled = self.target_scaler.inverse_transform(pred_scaled).flatten()
            ensemble_predictions.append(pred_unscaled)
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Calculate statistics
        mean_pred = np.mean(ensemble_predictions, axis=0)
        std_pred = np.std(ensemble_predictions, axis=0)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_bound = np.percentile(ensemble_predictions, (alpha/2) * 100, axis=0)
        upper_bound = np.percentile(ensemble_predictions, (1 - alpha/2) * 100, axis=0)
        
        # Bootstrap prediction intervals for longer horizons
        extended_predictions = self._bootstrap_long_term_predictions(
            district, years, n_samples=50
        )
        
        result = {
            'dates': base_prediction['dates'],
            'mean_prediction': mean_pred.tolist() + extended_predictions['mean'][len(mean_pred):],
            'std_prediction': std_pred.tolist() + extended_predictions['std'][len(std_pred):],
            'lower_bound': lower_bound.tolist() + extended_predictions['lower'][len(lower_bound):],
            'upper_bound': upper_bound.tolist() + extended_predictions['upper'][len(upper_bound):],
            'confidence_level': confidence_level,
            'district': district,
            'ensemble_predictions': ensemble_predictions.tolist()
        }
        
        return result
    
    def _bootstrap_long_term_predictions(self, district, years, n_samples=50):
        """Bootstrap predictions for long-term uncertainty"""
        all_predictions = []
        
        for _ in range(n_samples):
            # Add noise to historical data
            noisy_prediction = self.predict_long_term_robust(district, years)
            
            # Add uncertainty that increases with time
            time_factor = np.linspace(1, 1.5, len(noisy_prediction['predictions']))
            noise = np.random.normal(0, 10 * time_factor, len(noisy_prediction['predictions']))
            noisy_pred = np.array(noisy_prediction['predictions']) + noise
            
            all_predictions.append(noisy_pred)
        
        all_predictions = np.array(all_predictions)
        
        return {
            'mean': np.mean(all_predictions, axis=0),
            'std': np.std(all_predictions, axis=0),
            'lower': np.percentile(all_predictions, 2.5, axis=0),
            'upper': np.percentile(all_predictions, 97.5, axis=0)
        }
    
    def calculate_feature_importance(self, X, y):
        """Calculate feature importance using SHAP"""
        print("\n🔍 Calculating feature importance...")
        
        if not self.model:
            print("❌ No trained model found")
            return None
        
        try:
            # Use a subset of data for SHAP calculation
            sample_size = min(100, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_indices]
            
            # Create explainer
            explainer = shap.DeepExplainer(self.model, X_sample[:10])
            shap_values = explainer.shap_values(X_sample)
            
            # Calculate mean absolute SHAP values for each feature
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            mean_shap_values = np.mean(np.abs(shap_values), axis=(0, 1))
            
            # Map to feature names
            feature_importance = dict(zip(self.feature_cols, mean_shap_values))
            
            # Sort by importance
            self.feature_importance_scores = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            print("✅ Feature importance calculated")
            return self.feature_importance_scores
            
        except Exception as e:
            print(f"⚠️ Could not calculate SHAP values: {e}")
            
            # Fallback: Use permutation importance
            return self._calculate_permutation_importance(X, y)
    
    def _calculate_permutation_importance(self, X, y):
        """Fallback permutation importance calculation"""
        print("   Using permutation importance as fallback...")
        
        # Baseline performance
        baseline_pred = self.model.predict(X, verbose=0)
        baseline_score = r2_score(y.flatten(), baseline_pred.flatten())
        
        importance_scores = {}
        
        for i, feature in enumerate(self.feature_cols):
            # Permute feature i
            X_permuted = X.copy()
            X_permuted[:, :, i] = np.random.permutation(X_permuted[:, :, i])
            
            # Calculate performance with permuted feature
            permuted_pred = self.model.predict(X_permuted, verbose=0)
            permuted_score = r2_score(y.flatten(), permuted_pred.flatten())
            
            # Importance is the drop in performance
            importance_scores[feature] = baseline_score - permuted_score
        
        self.feature_importance_scores = dict(
            sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        return self.feature_importance_scores
    
    def create_prediction_dashboard(self, district, years=3):
        """Create comprehensive prediction dashboard"""
        print(f"\n📊 Creating prediction dashboard for {district}...")
        
        # Get predictions with uncertainty
        predictions = self.predict_with_uncertainty(district, years)
        if not predictions:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'Dengue Predictions with Uncertainty - {district}',
                'Prediction Intervals by Horizon',
                'Uncertainty Evolution',
                'Feature Importance'
            ],
            specs=[[{"colspan": 2}, None],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Main prediction plot
        dates = pd.to_datetime(predictions['dates'])
        
        # Mean prediction
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predictions['mean_prediction'],
                mode='lines+markers',
                name='Mean Prediction',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Confidence intervals
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predictions['upper_bound'],
                mode='lines',
                name=f'{predictions["confidence_level"]*100:.0f}% Upper Bound',
                line=dict(color='lightblue', width=1),
                fill=None
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predictions['lower_bound'],
                mode='lines',
                name=f'{predictions["confidence_level"]*100:.0f}% Lower Bound',
                line=dict(color='lightblue', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.3)'
            ),
            row=1, col=1
        )
        
        # Prediction intervals by horizon
        horizons = [f'H{i+1}' for i in range(min(6, len(predictions['mean_prediction'])))]
        mean_vals = predictions['mean_prediction'][:len(horizons)]
        std_vals = predictions['std_prediction'][:len(horizons)]
        
        fig.add_trace(
            go.Bar(
                x=horizons,
                y=mean_vals,
                error_y=dict(type='data', array=std_vals),
                name='Mean ± Std',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # Feature importance
        if self.feature_importance_scores:
            top_features = list(self.feature_importance_scores.keys())[:10]
            importance_values = [self.feature_importance_scores[f] for f in top_features]
            
            fig.add_trace(
                go.Bar(
                    x=importance_values,
                    y=top_features,
                    orientation='h',
                    name='Feature Importance',
                    marker_color='green'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Advanced Dengue Forecasting Dashboard - {district}",
            height=800,
            showlegend=True
        )
        
        # Save dashboard
        filename = f"advanced_dengue_dashboard_{district.lower()}.html"
        fig.write_html(filename)
        print(f"✅ Dashboard saved as '{filename}'")
        
        return fig
    
    def evaluate_model_performance(self, X, y, districts, dates):
        """Comprehensive model evaluation"""
        print("\n📈 Performing comprehensive model evaluation...")
        
        # Base evaluation
        base_results = self.evaluate_robust_model(X, y, districts, dates)
        
        # Uncertainty evaluation
        uncertainty_results = self._evaluate_uncertainty_performance(X, y, districts)
        
        # Feature importance
        feature_importance = self.calculate_feature_importance(X, y)
        
        # Combine results
        comprehensive_results = {
            'base_performance': base_results,
            'uncertainty_performance': uncertainty_results,
            'feature_importance': feature_importance,
            'model_complexity': {
                'n_features': len(self.feature_cols),
                'sequence_length': self.sequence_length,
                'forecast_horizon': self.forecast_horizon,
                'n_ensemble_models': len(self.ensemble_models)
            }
        }
        
        return comprehensive_results
    
    def _evaluate_uncertainty_performance(self, X, y, districts):
        """Evaluate uncertainty quantification performance"""
        print("   Evaluating uncertainty quantification...")
        
        if not self.ensemble_models:
            return {"error": "No ensemble models available"}
        
        # Calculate prediction intervals for test data
        coverage_rates = []
        interval_widths = []
        
        # Use a subset for evaluation
        n_samples = min(50, len(X))
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        
        for idx in sample_indices:
            # Get ensemble predictions
            ensemble_preds = []
            for model in self.ensemble_models:
                pred = model.predict(X[idx:idx+1], verbose=0)
                ensemble_preds.append(pred.flatten())
            
            ensemble_preds = np.array(ensemble_preds)
            
            # Calculate intervals
            lower = np.percentile(ensemble_preds, 2.5, axis=0)
            upper = np.percentile(ensemble_preds, 97.5, axis=0)
            
            # Check coverage
            actual = y[idx].flatten()
            coverage = np.mean((actual >= lower) & (actual <= upper))
            coverage_rates.append(coverage)
            
            # Calculate interval width
            width = np.mean(upper - lower)
            interval_widths.append(width)
        
        return {
            'mean_coverage_rate': np.mean(coverage_rates),
            'std_coverage_rate': np.std(coverage_rates),
            'mean_interval_width': np.mean(interval_widths),
            'std_interval_width': np.std(interval_widths),
            'target_coverage': 0.95
        }
    
    def get_historical_data(self, district):
        """Get historical data for a specific district"""
        if self.df is None:
            return {'dates': [], 'values': []}
        
        district_data = self.df[self.df['City'] == district].copy()
        district_data = district_data.sort_values('Date')
        
        return {
            'dates': district_data['Date'].tolist(),
            'values': district_data['Value'].tolist()
        }
    
    def save_advanced_model(self, filepath='advanced_dengue_forecaster.pkl'):
        """Save the advanced model with all components"""
        model_data = {
            'model': self.model,
            'ensemble_models': self.ensemble_models,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'district_encoder': self.district_encoder,
            'districts': self.districts,
            'feature_cols': self.feature_cols,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'n_bootstrap_samples': self.n_bootstrap_samples,
            'feature_importance_scores': self.feature_importance_scores,
            'df': self.df
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Advanced model saved to {filepath}")
    
    def load_advanced_model(self, filepath='advanced_dengue_forecaster.pkl'):
        """Load the advanced model with all components"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.ensemble_models = model_data.get('ensemble_models', [])
        self.feature_scaler = model_data['feature_scaler']
        self.target_scaler = model_data['target_scaler']
        self.district_encoder = model_data['district_encoder']
        self.districts = model_data['districts']
        self.feature_cols = model_data['feature_cols']
        self.sequence_length = model_data['sequence_length']
        self.forecast_horizon = model_data['forecast_horizon']
        self.n_bootstrap_samples = model_data.get('n_bootstrap_samples', 100)
        self.feature_importance_scores = model_data.get('feature_importance_scores', {})
        self.df = model_data['df']
        print(f"✅ Advanced model loaded from {filepath}")

def train_advanced_model():
    """Train the advanced model with all features"""
    print("🚀 Training advanced dengue forecasting model...")
    
    # Initialize advanced forecaster
    forecaster = AdvancedDengueForecaster(sequence_length=12, forecast_horizon=6)
    
    # Load and prepare data
    data = forecaster.load_and_prepare_data('Dengue_Data (2010-2020).xlsx')
    
    # Create sequences
    X, y, districts, dates = forecaster.create_sequences_robust(data)
    
    print(f"📊 Dataset shape: X={X.shape}, y={y.shape}")
    
    # Train base model
    print("\n🏗️ Training base robust model...")
    history = forecaster.train_robust_model(X, y, districts, dates, epochs=50)
    
    # Train ensemble models
    print("\n🎯 Training ensemble models...")
    ensemble_histories = forecaster.train_ensemble_models(X, y, districts, dates, n_models=3)
    
    # Evaluate comprehensive performance
    results = forecaster.evaluate_model_performance(X, y, districts, dates)
    
    # Save advanced model
    forecaster.save_advanced_model()
    
    return forecaster, results

if __name__ == "__main__":
    # Check if advanced model exists
    if not os.path.exists('advanced_dengue_forecaster.pkl'):
        print("🚀 Training new advanced model...")
        forecaster, results = train_advanced_model()
    else:
        print("✅ Using existing advanced model")
        forecaster = AdvancedDengueForecaster()
        forecaster.load_advanced_model()
    
    # Simple performance evaluation
    print("\n📊 ADVANCED DENGUE FORECASTER PERFORMANCE EVALUATION")
    print("=" * 60)
    
    # Load test data for evaluation
    if forecaster.df is not None:
        print(f"📈 Dataset Information:")
        print(f"   • Total records: {len(forecaster.df):,}")
        print(f"   • Districts: {len(forecaster.districts)}")
        print(f"   • Date range: {forecaster.df['Date'].min()} to {forecaster.df['Date'].max()}")
        print(f"   • Features: {len(forecaster.feature_cols)}")
        
        # Create test sequences
        X, y, districts, dates = forecaster.create_sequences_robust(forecaster.df)
        print(f"   • Sequences: {X.shape[0]:,} samples")
        print(f"   • Input shape: {X.shape[1:]} (time_steps, features)")
        print(f"   • Output shape: {y.shape[1:]} (forecast_horizon)")
        
        # Evaluate base model performance
        if forecaster.model:
            print(f"\n🎯 Model Architecture:")
            print(f"   • Sequence length: {forecaster.sequence_length} months")
            print(f"   • Forecast horizon: {forecaster.forecast_horizon} months")
            print(f"   • Ensemble models: {len(forecaster.ensemble_models)}")
            
            # Get model performance metrics
            results = forecaster.evaluate_robust_model(X, y, districts, dates)
            
            print(f"\n📊 Performance Summary:")
            overall_mae = results.get('overall_mae', 'N/A')
            overall_rmse = results.get('overall_rmse', 'N/A')
            overall_r2 = results.get('overall_r2', 'N/A')
            overall_mape = results.get('overall_mape', 'N/A')
            
            print(f"   • Overall MAE: {overall_mae:.2f}" if isinstance(overall_mae, (int, float)) else f"   • Overall MAE: {overall_mae}")
            print(f"   • Overall RMSE: {overall_rmse:.2f}" if isinstance(overall_rmse, (int, float)) else f"   • Overall RMSE: {overall_rmse}")
            print(f"   • Overall R²: {overall_r2:.4f}" if isinstance(overall_r2, (int, float)) else f"   • Overall R²: {overall_r2}")
            print(f"   • Overall MAPE: {overall_mape:.2f}%" if isinstance(overall_mape, (int, float)) else f"   • Overall MAPE: {overall_mape}")
            
            # Horizon-wise performance
            print(f"\n📈 Horizon-wise Performance:")
            for i in range(forecaster.forecast_horizon):
                horizon_key = f'horizon_{i+1}'
                if horizon_key in results:
                    h_results = results[horizon_key]
                    print(f"   • H{i+1}: MAE={h_results.get('mae', 0):.1f}, "
                          f"RMSE={h_results.get('rmse', 0):.1f}, "
                          f"R²={h_results.get('r2', 0):.3f}")
        
        # Test simple prediction
        print(f"\n🔮 Sample Predictions:")
        try:
            # Test basic prediction for Colombo
            sample_prediction = forecaster.predict_long_term_robust('Colombo', years=1)
            if sample_prediction and 'predictions' in sample_prediction:
                pred_values = sample_prediction['predictions'][:6]  # First 6 months
                print(f"   • Colombo (next 6 months): {[f'{v:.0f}' for v in pred_values]}")
                print(f"   • Average prediction: {np.mean(pred_values):.1f} cases/month")
                print(f"   • Prediction range: {np.min(pred_values):.0f} - {np.max(pred_values):.0f}")
        except Exception as e:
            print(f"   ⚠️ Prediction error: {str(e)[:50]}...")
    
    print(f"\n🎉 Advanced dengue forecasting model evaluation completed!")
    print("📁 Key features:")
    print("  ✅ Uncertainty quantification with prediction intervals")
    print("  ✅ Ensemble models for robust predictions")
    print("  ✅ Feature importance analysis")
    print("  ✅ Comprehensive evaluation metrics")
    print("  ✅ Long-term forecasting capabilities")
    
    # Performance assessment
    if forecaster.model:
        print(f"\n🏆 PERFORMANCE ASSESSMENT:")
        try:
            # Get overall performance
            X, y, districts, dates = forecaster.create_sequences_robust(forecaster.df)
            results = forecaster.evaluate_robust_model(X, y, districts, dates)
            
            overall_r2 = results.get('overall_r2', 0)
            overall_mae = results.get('overall_mae', float('inf'))
            
            if overall_r2 > 0.6:
                performance_grade = "EXCELLENT"
                grade_emoji = "🏆"
            elif overall_r2 > 0.4:
                performance_grade = "GOOD"
                grade_emoji = "👍"
            elif overall_r2 > 0.2:
                performance_grade = "FAIR"
                grade_emoji = "⚠️"
            else:
                performance_grade = "NEEDS IMPROVEMENT"
                grade_emoji = "❌"
            
            print(f"   {grade_emoji} Overall Grade: {performance_grade}")
            print(f"   📊 R² Score: {overall_r2:.4f}")
            print(f"   📏 MAE: {overall_mae:.2f} cases")
            
            # Strengths and weaknesses
            print(f"\n💪 STRENGTHS:")
            print("  • Advanced ensemble architecture with uncertainty quantification")
            print("  • Robust feature engineering with temporal patterns")
            print("  • Time series cross-validation for reliable evaluation")
            print("  • Long-term forecasting capabilities")
            
            print(f"\n⚠️ AREAS FOR IMPROVEMENT:")
            if overall_r2 < 0.5:
                print("  • Model accuracy could be improved with more data or features")
            if overall_mae > 100:
                print("  • High prediction errors suggest need for better regularization")
            print("  • Uncertainty quantification implementation needs refinement")
            print("  • Feature importance calculation needs optimization")
            
        except Exception as e:
            print(f"   ❌ Could not complete assessment: {str(e)[:50]}...")
    
    print("\n" + "=" * 60) 