#!/usr/bin/env python3
"""
Script to run the Dengue Forecasting Dashboard
"""

import subprocess
import sys
import os

def check_model_exists():
    """Check if the trained model exists"""
    if not os.path.exists('enhanced_dengue_forecaster.pkl'):
        print("❌ Trained model not found!")
        print("🔄 Training the model first...")
        
        # Run the training script
        try:
            subprocess.run([sys.executable, 'dengue_predictor_enhanced.py'], check=True)
            print("✅ Model training completed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error training model: {e}")
            return False
    else:
        print("✅ Trained model found!")
    
    return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Dengue Forecasting Dashboard...")
    print("🌐 The dashboard will open in your web browser")
    print("📍 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'dengue_dashboard.py',
            '--server.port=8501',
            '--server.address=localhost',
            '--browser.gatherUsageStats=false'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")

if __name__ == "__main__":
    print("🦟 DENGUE FORECASTING DASHBOARD LAUNCHER")
    print("=" * 50)
    
    # Check if model exists, train if not
    if check_model_exists():
        # Run the dashboard
        run_dashboard()
    else:
        print("❌ Cannot start dashboard without trained model")
        sys.exit(1) 