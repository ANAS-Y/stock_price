import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os 
from copy import deepcopy 
from sklearn.preprocessing import MinMaxScaler

# --- 1. CONFIGURATION AND LOADING ---

# Define file paths
MODEL_FILE = 'best_rnn_model.h5' 
DATA_FILE = 'cleaned_nigerian_stock_data.csv' 
LOOKBACK_PERIOD = 60
TRAINING_FEATURES = ['Price', 'Open', 'High', 'Low', 'Change %']
MAX_DAILY_CHANGE = 0.05 

st.set_page_config(layout="wide", page_title="Stock Price Prediction")

# Custom CSS for a more attractive interface
st.markdown("""
<style>
    .stApp {
        background-color:yellow;
    }
    .main-header {
        font-size: 2.5rem;
        color: #007A33; /* NSE Green */
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
        st.error(f"Missing files. Ensure {MODEL_FILE} and {DATA_FILE} are present.")
        return None, None

    try:
        model = load_model(MODEL_FILE, compile=False)
        df = pd.read_csv(DATA_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Robust Cleaning
        for col in TRAINING_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
                
        # Volume Cleaning
        if 'Vol.' in df.columns:
            def clean_vol_helper(x):
                if isinstance(x, (int, float)): return x
                x = str(x).upper().replace(',', '').strip()
                if x == '-' or x == 'NAN': return 0.0
                if 'M' in x: return float(x.replace('M', '')) * 1_000_000
                if 'K' in x: return float(x.replace('K', '')) * 1_000
                if 'B' in x: return float(x.replace('B', '')) * 1_000_000_000
                try: return float(x)
                except: return 0.0
            
            if df['Vol.'].dtype == 'object':
                df['Vol.'] = df['Vol.'].apply(clean_vol_helper)
            df['Vol.'] = pd.to_numeric(df['Vol.'], errors='coerce').fillna(0.0)

        return model, df
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

model, df_all = load_resources()

# --- 2. SIDEBAR CONFIGURATION ---

# Custom Header
st.markdown('<div class="main-header">🇳🇬 NSE Stock Price Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Recurrent Neural Network (RNN) Strategy</div>', unsafe_allow_html=True)

if df_all is None or model is None:
    st.stop()

with st.sidebar:
    st.header("⚙️ Configuration")
    organizations = df_all['Organisation'].unique()
    default_idx = int(np.where(organizations == 'NSE')[0][0]) if 'NSE' in organizations else 0
    selected_org = st.selectbox("Select Ticker:", organizations, index=default_idx)

    df_org = df_all[df_all['Organisation'] == selected_org].sort_values('Date').copy()
    last_available_date = df_org['Date'].max()

    st.markdown("---")
    st.subheader("📅 Prediction Settings")

    min_date = last_available_date + pd.Timedelta(days=1)
    default_date = max(min_date.date(), datetime.now().date())

    prediction_date = st.date_input(
        "Target Date:",
        value=default_date,
        min_value=min_date.date(),
        max_value=min_date.date() + timedelta(days=730)
    )
    prediction_date = pd.to_datetime(prediction_date)

    # Model Info
    model_name = getattr(model, 'name', 'RNN Model')
    st.info(f"**Model:** {model_name}\n\n**Strategy:** Localized Scaling with Volatility Constraints")

# --- 3. RECURSIVE PREDICTION LOGIC ---

def predict_recursive(org_data, model, target_date):
    if len(org_data) < LOOKBACK_PERIOD:
        return None, None

    # 1. LOCAL SCALING
    local_scaler = MinMaxScaler(feature_range=(0, 1))
    data_values = org_data[TRAINING_FEATURES].values
    local_scaler.fit(data_values)
    
    # 2. Initial Sequence
    last_60 = data_values[-LOOKBACK_PERIOD:]
    current_seq = local_scaler.transform(last_60)
    
    predictions = []
    dates = []
    curr_date = last_available_date + timedelta(days=1)
    
    PRICE_IDX = TRAINING_FEATURES.index('Price')
    last_price = data_values[-1, PRICE_IDX] 

    while curr_date <= target_date:
        if curr_date.weekday() < 5: 
            input_tensor = current_seq.reshape(1, LOOKBACK_PERIOD, len(TRAINING_FEATURES))
            pred_scaled = model.predict(input_tensor, verbose=0)[0, 0]
            
            dummy = np.zeros((1, len(TRAINING_FEATURES)))
            dummy[0, PRICE_IDX] = pred_scaled
            pred_unscaled = local_scaler.inverse_transform(dummy)[0, PRICE_IDX]
            
            # Stability Constraint
            baseline = predictions[-1] if predictions else last_price
            min_p = baseline * (1 - MAX_DAILY_CHANGE)
            max_p = baseline * (1 + MAX_DAILY_CHANGE)
            pred_constrained = np.clip(pred_unscaled, min_p, max_p)
            if pred_constrained <= 0: pred_constrained = 0.01
            
            predictions.append(pred_constrained)
            dates.append(curr_date)
            
            # Update Sequence
            next_step_unscaled = np.full((1, len(TRAINING_FEATURES)), pred_constrained)
            next_step_unscaled[0, TRAINING_FEATURES.index('Change %')] = 0.0
            next_step_scaled = local_scaler.transform(next_step_unscaled)
            current_seq = np.vstack([current_seq[1:], next_step_scaled])
            
        curr_date += timedelta(days=1)
        
    if not predictions: return None, None
    
    df_pred = pd.DataFrame({'Date': dates, 'Price': predictions})
    return predictions[-1], df_pred

# --- 4. DASHBOARD DISPLAY ---

# Current Status Section
st.subheader(f"📊 Market Status: {selected_org}")
col1, col2, col3 = st.columns(3)

latest_close = df_org['Price'].iloc[-1]
latest_date_str = last_available_date.strftime('%d %b, %Y')
avg_vol = df_org['Vol.'].mean()

with col1:
    st.metric(label="Last Close Price", value=f"₦{latest_close:,.2f}", delta=None)
    st.caption(f"As of {latest_date_str}")

with col2:
    st.metric(label="Average Volume", value=f"{avg_vol/1e6:.2f}M")

with col3:
    total_data_points = len(df_org)
    st.metric(label="Data Points", value=f"{total_data_points}")

# Forecast Section
st.divider()
st.subheader(f"📈 Forecast Analysis")

if st.button("Generate Forecast", type="primary", use_container_width=True):
    with st.spinner(f"Running Advanced Analysis for {selected_org}..."):
        final_price, df_path = predict_recursive(df_org, model, prediction_date)
        
    if final_price is not None:
        # --- NEW: Calculate Percentage Change ---
        price_change = final_price - latest_close
        percent_change = (price_change / latest_close) * 100
        
        # Display Key Forecast Metrics
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            st.metric(
                label=f"Predicted Price ({prediction_date.strftime('%d %b, %Y')})", 
                value=f"₦{final_price:,.2f}", 
                delta=f"{price_change:,.2f} ({percent_change:+.2f}%)"
            )
        
        with m_col2:
            trend_icon = "↗️ UP" if percent_change > 0 else "↘️ DOWN"
            color = "green" if percent_change > 0 else "red"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color:{color}; margin:0;">{trend_icon}</h3>
                <p style="margin:0;">Predicted Trend</p>
            </div>
            """, unsafe_allow_html=True)

        # Enhanced Plotting
        st.markdown("### Price Trajectory")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot only last 365 days of history for clarity
        history_plot = df_org.tail(365)
        
        ax.plot(history_plot['Date'], history_plot['Price'], label='Historical (Last 1 Year)', color='#555555', linewidth=1.5, alpha=0.7)
        ax.plot(df_path['Date'], df_path['Price'], label='Forecast', color='#007A33', linewidth=2.5, linestyle='-') # NSE Green
        
        # Highlight End Point
        ax.scatter(df_path['Date'].iloc[-1], final_price, color='#FF4B4B', s=150, zorder=5, edgecolors='white', linewidth=2)
        
        # Styling the Chart
        ax.set_title(f"{selected_org} Price Projection", fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel("Price (₦)", fontsize=12)
        ax.grid(True, which='major', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left', frameon=True)
        
        st.pyplot(fig)
        
        # Data Table
        with st.expander("View Detailed Forecast Data"):
            st.dataframe(df_path.style.format({"Price": "₦{:,.2f}"}))
            
    else:
        st.error("Insufficient data for forecasting.")