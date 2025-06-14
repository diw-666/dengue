import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from advanced_dengue_forecaster import AdvancedDengueForecaster, train_advanced_model
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🦟 Dengue Forecasting Dashboard - Sri Lanka",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e !important;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        color: white !important;
    }
    .prediction-info {
        background-color: #1a1a1a;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #1f77b4;
        color: white !important;
    }
    
    /* Dark theme for main content */
    .main .block-container {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Dark theme for sidebar */
    .css-1d391kg {
        background-color: #1a1a1a !important;
    }
    
    /* White text for all elements */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, div, span {
        color: #ffffff !important;
    }
    
    /* Dark theme for metrics */
    .metric-container {
        background-color: #2d2d2d !important;
        border: 1px solid #444444;
        border-radius: 5px;
        padding: 10px;
        color: white !important;
    }
    
    /* Dark theme for selectbox and slider */
    .stSelectbox label, .stSlider label {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    
    .stSelectbox > div > div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* Dark theme for buttons */
    .stButton button {
        background-color: #1f77b4 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
    }
    
    /* Dark theme for info boxes */
    .stInfo {
        background-color: #1a1a1a !important;
        border: 1px solid #1f77b4 !important;
        color: white !important;
    }
    
    /* Dark theme for success messages */
    .stSuccess {
        background-color: #1a4d1a !important;
        border: 1px solid #28a745 !important;
        color: #ffffff !important;
    }
    
    /* Force dark background for entire app */
    .stApp {
        background-color: #000000 !important;
    }
    
    /* Dark theme for sidebar elements */
    .css-1d391kg .stMarkdown, .css-1d391kg p, .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #ffffff !important;
    }
    
    /* Force visibility for all elements */
    * {
        opacity: 1 !important;
    }
    
    /* Ensure proper text rendering with dark theme */
    body, .main, .sidebar {
        font-family: "Source Sans Pro", sans-serif !important;
        color: #ffffff !important;
        background-color: #000000 !important;
    }
    
    /* Dark theme for download button */
    .stDownloadButton button {
        background-color: #28a745 !important;
        color: white !important;
    }
    
    /* Dark theme for columns */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: transparent !important;
    }
</style>

<script>
// Force dark theme after page load
setTimeout(function() {
    document.body.style.backgroundColor = '#000000';
    document.body.style.color = '#ffffff';
    document.body.style.opacity = '1';
    document.body.style.visibility = 'visible';
    
    // Force redraw with dark theme
    var elements = document.querySelectorAll('*');
    elements.forEach(function(el) {
        el.style.opacity = '1';
        el.style.visibility = 'visible';
    });
    
    // Force dark background for main container
    var mainContainer = document.querySelector('.main');
    if (mainContainer) {
        mainContainer.style.backgroundColor = '#000000';
        mainContainer.style.color = '#ffffff';
    }
}, 100);
</script>
""", unsafe_allow_html=True)

@st.cache_resource
def load_forecaster():
    """Load the trained forecaster model"""
    try:
        if not os.path.exists('advanced_dengue_forecaster.pkl'):
            st.warning("🔄 Model not found! Training new model... This may take a few minutes.")
            
            # Import the training function
            from advanced_dengue_forecaster import train_advanced_model
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🧠 Training AI model... Please wait (this may take 2-3 minutes)"):
                status_text.text("📊 Loading and preparing data...")
                progress_bar.progress(20)
                
                try:
                    # Train the model
                    status_text.text("🔥 Training neural network...")
                    progress_bar.progress(50)
                    
                    forecaster = train_advanced_model()
                    
                    progress_bar.progress(90)
                    status_text.text("✅ Model training completed!")
                    progress_bar.progress(100)
                    
                    # Clear the progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("🎉 Model trained successfully!")
                    return forecaster
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {e}")
                    st.error("Please check if the data file 'Dengue_Data (2010-2020).xlsx' is present.")
                    st.stop()
        else:
            # Load existing model
            forecaster = AdvancedDengueForecaster()
            forecaster.load_advanced_model()
            return forecaster
            
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

@st.cache_data
def get_district_list(_forecaster):
    """Get list of available districts"""
    try:
        if hasattr(_forecaster, 'districts') and _forecaster.districts is not None:
            return sorted(_forecaster.districts.tolist())
        elif hasattr(_forecaster, 'df') and _forecaster.df is not None:
            return sorted(_forecaster.df['City'].unique().tolist())
        else:
            return ['Colombo', 'Gampaha', 'Kalutara']  # Default fallback
    except Exception as e:
        return ['Colombo', 'Gampaha', 'Kalutara']  # Default fallback

@st.cache_data
def get_historical_data(_forecaster, district):
    """Get historical data for a district"""
    try:
        return _forecaster.get_historical_data(district)
    except Exception as e:
        # Fallback: get data directly from the dataframe
        if hasattr(_forecaster, 'df') and _forecaster.df is not None:
            district_data = _forecaster.df[_forecaster.df['City'] == district].copy()
            district_data = district_data.sort_values('Date')
            return {
                'dates': district_data['Date'].tolist(),
                'values': district_data['Value'].tolist()
            }
        else:
            return {'dates': [], 'values': []}

def create_prediction_chart(historical_data, prediction_data, district):
    """Create an interactive prediction chart"""
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Historical Data and 5-Year Predictions for {district}',
            'Monthly Seasonal Pattern'
        ),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data['dates'],
            y=historical_data['values'],
            mode='lines+markers',
            name='Historical Cases',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Date:</b> %{x}<br><b>Cases:</b> %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Predictions
    fig.add_trace(
        go.Scatter(
            x=prediction_data['dates'],
            y=prediction_data['predictions'],
            mode='lines+markers',
            name='Predicted Cases',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=4),
            hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Cases:</b> %{y:.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add confidence interval (simple approach using std)
    pred_std = np.std(prediction_data['predictions'])
    upper_bound = np.array(prediction_data['predictions']) + 1.96 * pred_std
    lower_bound = np.array(prediction_data['predictions']) - 1.96 * pred_std
    lower_bound = np.maximum(lower_bound, 0)  # Ensure non-negative
    
    fig.add_trace(
        go.Scatter(
            x=prediction_data['dates'] + prediction_data['dates'][::-1],
            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # Monthly seasonal pattern
    pred_df = pd.DataFrame({
        'date': prediction_data['dates'],
        'cases': prediction_data['predictions']
    })
    pred_df['month'] = pd.to_datetime(pred_df['date']).dt.month
    monthly_avg = pred_df.groupby('month')['cases'].mean()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig.add_trace(
        go.Bar(
            x=months,
            y=monthly_avg.values,
            name='Average Monthly Cases',
            marker_color='#2ca02c',
            hovertemplate='<b>Month:</b> %{x}<br><b>Avg Cases:</b> %{y:.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Dengue Forecasting Dashboard - {district}",
        title_x=0.5,
        hovermode='x unified'
    )
    
    # Update x-axis
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    
    # Update y-axis
    fig.update_yaxes(title_text="Number of Cases", row=1, col=1)
    fig.update_yaxes(title_text="Average Cases", row=2, col=1)
    
    return fig

def create_summary_metrics(historical_data, prediction_data):
    """Create summary metrics"""
    hist_avg = np.mean(historical_data['values'])
    hist_max = np.max(historical_data['values'])
    hist_min = np.min(historical_data['values'])
    
    pred_avg = np.mean(prediction_data['predictions'])
    pred_max = np.max(prediction_data['predictions'])
    pred_min = np.min(prediction_data['predictions'])
    
    # Calculate trend
    trend = "📈 Increasing" if pred_avg > hist_avg else "📉 Decreasing"
    change_pct = ((pred_avg - hist_avg) / hist_avg) * 100
    
    return {
        'historical': {'avg': hist_avg, 'max': hist_max, 'min': hist_min},
        'predicted': {'avg': pred_avg, 'max': pred_max, 'min': pred_min},
        'trend': trend,
        'change_pct': change_pct
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🦟 Dengue Forecasting Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #ffffff !important; font-weight: bold;">Sri Lanka - Neural Network Predictions</h2>', unsafe_allow_html=True)
    
    # Add some spacing
    st.markdown("---")
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        forecaster = load_forecaster()
    
    st.success("✅ AI Model loaded successfully!")
    st.markdown("")  # Add spacing
    
    # Create main layout with sidebar and content
    # Sidebar
    st.sidebar.markdown("# 🎛️ Control Panel")
    st.sidebar.markdown("")
    
    # District selection with better formatting
    st.sidebar.markdown("### 📍 Location Selection")
    districts = get_district_list(forecaster)
    selected_district = st.sidebar.selectbox(
        "Choose District:",
        districts,
        index=districts.index('Colombo') if 'Colombo' in districts else 0,
        help="Select any district in Sri Lanka for prediction"
    )
    
    st.sidebar.markdown("")
    
    # Prediction years with better formatting
    st.sidebar.markdown("### 📅 Forecast Settings")
    prediction_years = st.sidebar.slider(
        "Prediction Period (Years):",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Choose how many years ahead to predict"
    )
    
    # Quick prediction buttons
    st.sidebar.markdown("### ⚡ Quick Predictions")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📊 3 Years", key="quick_3"):
            prediction_years = 3
    with col2:
        if st.button("📈 10 Years", key="quick_10"):
            prediction_years = 10
    
    st.sidebar.markdown("---")
    
    # Model information in expandable section
    with st.sidebar.expander("🤖 Model Information", expanded=False):
        st.markdown(f"""
        **Architecture:** LSTM + Attention Neural Network  
        **Input Sequence:** {forecaster.sequence_length} months  
        **Features Used:** {len(forecaster.feature_cols)} variables  
        **Districts Covered:** {len(forecaster.districts)} locations  
        **Training Data:** 2010-2020 (11 years)  
        **Forecast Method:** Multi-step iterative prediction
        """)
    
    # Add refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Dashboard", help="Refresh the dashboard if content appears faded"):
        st.rerun()
    
    # Main content area
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📈 Analysis", "💾 Export"])
    
    with tab1:
        # Prediction section
        st.markdown(f"## 🔮 Forecasting for {selected_district}")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Selected District:** {selected_district}")
            st.markdown(f"**Forecast Period:** {prediction_years} years ({prediction_years * 12} months)")
        
        with col2:
            if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
                st.rerun()
        
        with col3:
            st.markdown("**Status:** Ready")
        
        st.markdown("---")
        
        # Generate predictions
        with st.spinner(f"🔮 Generating {prediction_years}-year predictions for {selected_district}..."):
            try:
                # Get historical data
                historical_data = get_historical_data(forecaster, selected_district)
                
                # Generate predictions
                prediction_data = forecaster.predict_long_term_robust(selected_district, years=prediction_years)
                
                # Create metrics
                metrics = create_summary_metrics(historical_data, prediction_data)
                
                # Display key metrics in a nice layout
                st.markdown("### 📊 Key Performance Indicators")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "📈 Historical Avg",
                        f"{metrics['historical']['avg']:.0f}",
                        help="Average monthly cases from 2010-2020"
                    )
                
                with col2:
                    st.metric(
                        "🔮 Predicted Avg",
                        f"{metrics['predicted']['avg']:.0f}",
                        delta=f"{metrics['change_pct']:+.1f}%",
                        help="Average monthly cases for prediction period"
                    )
                
                with col3:
                    st.metric(
                        "⚠️ Peak Prediction",
                        f"{metrics['predicted']['max']:.0f}",
                        help="Highest predicted monthly cases"
                    )
                
                with col4:
                    trend_emoji = "📈" if "Increasing" in metrics['trend'] else "📉"
                    st.metric(
                        "📊 Trend",
                        f"{trend_emoji} {metrics['trend'].split()[1]}",
                        help="Overall trend compared to historical average"
                    )
                
                st.markdown("")
                
                # Prediction summary card
                st.markdown(f"""
                <div class="prediction-info">
                    <h4>🎯 Prediction Summary</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <p><strong>🏢 District:</strong> {selected_district}</p>
                            <p><strong>📅 Period:</strong> {prediction_data['dates'][0].strftime('%B %Y')} to {prediction_data['dates'][-1].strftime('%B %Y')}</p>
                        </div>
                        <div>
                            <p><strong>📊 Total Months:</strong> {len(prediction_data['predictions'])}</p>
                            <p><strong>🎯 Confidence:</strong> Based on {len(historical_data['dates'])} months of data</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("")
                
                # Create and display chart
                fig = create_prediction_chart(historical_data, prediction_data, selected_district)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                st.info("💡 Make sure the model is properly trained and the district name is correct.")
    
    with tab2:
        # Analysis section
        st.markdown("## 📈 Detailed Analysis")
        
        if 'prediction_data' in locals():
            # Create two columns for analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📅 Yearly Breakdown")
                
                # Create yearly breakdown
                pred_df = pd.DataFrame({
                    'date': prediction_data['dates'],
                    'cases': prediction_data['predictions']
                })
                pred_df['year'] = pd.to_datetime(pred_df['date']).dt.year
                yearly_totals = pred_df.groupby('year')['cases'].sum().round(0)
                
                # Display as a nice table
                yearly_data = []
                for year, total in yearly_totals.items():
                    yearly_data.append({"Year": year, "Predicted Cases": f"{total:.0f}", "Monthly Avg": f"{total/12:.0f}"})
                
                st.dataframe(pd.DataFrame(yearly_data), use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Risk Assessment")
                
                # Risk levels based on predicted values
                high_risk_months = sum(1 for x in prediction_data['predictions'] if x > metrics['historical']['avg'] * 1.5)
                moderate_risk_months = sum(1 for x in prediction_data['predictions'] if metrics['historical']['avg'] < x <= metrics['historical']['avg'] * 1.5)
                low_risk_months = len(prediction_data['predictions']) - high_risk_months - moderate_risk_months
                
                # Create risk assessment chart
                risk_data = pd.DataFrame({
                    'Risk Level': ['🔴 High Risk', '🟡 Moderate Risk', '🟢 Low Risk'],
                    'Months': [high_risk_months, moderate_risk_months, low_risk_months],
                    'Percentage': [
                        f"{(high_risk_months/len(prediction_data['predictions'])*100):.1f}%",
                        f"{(moderate_risk_months/len(prediction_data['predictions'])*100):.1f}%",
                        f"{(low_risk_months/len(prediction_data['predictions'])*100):.1f}%"
                    ]
                })
                
                st.dataframe(risk_data, use_container_width=True)
            
            # Seasonal analysis
            st.markdown("### 🌡️ Seasonal Pattern Analysis")
            
            pred_df['month'] = pd.to_datetime(pred_df['date']).dt.month
            pred_df['month_name'] = pd.to_datetime(pred_df['date']).dt.month_name()
            monthly_stats = pred_df.groupby(['month', 'month_name'])['cases'].agg(['mean', 'max', 'min']).round(0)
            monthly_stats = monthly_stats.reset_index()
            monthly_stats.columns = ['Month #', 'Month', 'Average Cases', 'Peak Cases', 'Minimum Cases']
            
            st.dataframe(monthly_stats[['Month', 'Average Cases', 'Peak Cases', 'Minimum Cases']], use_container_width=True)
            
        else:
            st.info("📊 Generate predictions first to see detailed analysis")
    
    with tab3:
        # Export section
        st.markdown("## 💾 Data Export & Download")
        
        if 'prediction_data' in locals():
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📥 Download Options")
                
                # Prepare download data
                download_df = pd.DataFrame({
                    'Date': prediction_data['dates'],
                    'District': selected_district,
                    'Predicted_Cases': [round(x, 0) for x in prediction_data['predictions']],
                    'Year': [d.year for d in prediction_data['dates']],
                    'Month': [d.month for d in prediction_data['dates']],
                    'Month_Name': [d.strftime('%B') for d in prediction_data['dates']]
                })
                
                csv = download_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"dengue_predictions_{selected_district}_{prediction_years}years.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Summary report
                summary_report = f"""
                DENGUE FORECASTING REPORT
                ========================
                
                District: {selected_district}
                Prediction Period: {prediction_data['dates'][0].strftime('%B %Y')} to {prediction_data['dates'][-1].strftime('%B %Y')}
                Total Months: {len(prediction_data['predictions'])}
                
                SUMMARY STATISTICS:
                - Historical Average: {metrics['historical']['avg']:.0f} cases/month
                - Predicted Average: {metrics['predicted']['avg']:.0f} cases/month
                - Predicted Maximum: {metrics['predicted']['max']:.0f} cases/month
                - Predicted Minimum: {metrics['predicted']['min']:.0f} cases/month
                - Trend: {metrics['trend']}
                - Change: {metrics['change_pct']:+.1f}%
                
                RISK ASSESSMENT:
                - High Risk Months: {high_risk_months}
                - Moderate Risk Months: {moderate_risk_months}
                - Low Risk Months: {low_risk_months}
                
                Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                Model: LSTM + Attention Neural Network
                """
                
                st.download_button(
                    label="📄 Download Summary Report (TXT)",
                    data=summary_report,
                    file_name=f"dengue_report_{selected_district}_{prediction_years}years.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("### 📊 Data Preview")
                st.markdown("**First 10 predictions:**")
                st.dataframe(download_df.head(10), use_container_width=True)
                
                st.markdown(f"**Total records:** {len(download_df)}")
                st.markdown(f"**File size:** ~{len(csv)/1024:.1f} KB")
        
        else:
            st.info("📊 Generate predictions first to enable data export")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #ffffff; padding: 2rem;">
        <p>🦟 <strong>Dengue Forecasting Dashboard</strong> | Powered by Neural Networks & AI</p>
        <p>📊 Data: Sri Lanka Dengue Cases (2010-2020) | 🤖 Model: LSTM + Attention Mechanism</p>
        <p>⚠️ <em>Predictions are for research purposes. Consult health authorities for official guidance.</em></p>
        <p>🔬 <strong>Developed for Public Health Research & Planning</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 