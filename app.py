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

st.set_page_config(layout="wide", page_title="NSE Stock Predictor", page_icon="📈")

# --- 2. CUSTOM CSS FOR DARK THEME UI ---
st.markdown("""
<style>
    /* Main Background - Dark Financial Theme */
    .stApp {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364); /* "203a43" is a dark slate green */
        color: #ffffff;
    }
    
    /* Main Title Styling - Now White for Contrast */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem;
        color: #ffffff; 
        font-weight: 800;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 1rem;
        text-shadow: 0px 0px 10px rgba(0,0,0,0.5);
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #e0e0e0; /* Light Gray */
        text-align: center;
        margin-bottom: 2.5rem;
        font-style: italic;
    }

    /* Card Containers for Metrics - White Cards on Dark Background */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: none;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3); /* Stronger shadow for depth */
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        text-align: center;
        color: #333; /* Text inside card remains dark */
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 25px rgba(0,0,0,0.4);
        border: 1px solid #007A33;
    }
    
    /* Force label colors inside metrics to be dark (since card is white) */
    div[data-testid="stMetric"] label {
        color: #555555 !important;
        font-weight: bold;
    }
    
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #000000 !important;
    }

    /* Custom Card for Trend Indicator */
    .trend-card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(45deg, #00b09b, #96c93d); /* Brighter Green Gradient */
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        width: 100%;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(45deg, #96c93d, #00b09b);
        box-shadow: 0 6px 20px rgba(0, 255, 100, 0.4); /* Glowing effect */
        transform: scale(1.02);
        color: #ffffff;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa; /* Keep sidebar light for contrast */
        border-right: 1px solid #eaeaea;
    }
    
    /* Sidebar Text Fix */
    section[data-testid="stSidebar"] * {
        color: #333333;
    }

    /* Chart Container Styling */
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    
    /* Divider Line Color */
    hr {
        border-color: #555;
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

# --- 3. UI HEADER ---

st.markdown('<div class="main-header">🇳🇬 Stock Price Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Stock Price Prediction using LSTM RNN </div>', unsafe_allow_html=True)

if df_all is None or model is None:
    st.warning("Data or Model not found. Please verify deployment files.")
    st.stop()

# --- 4. SIDEBAR ---

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Nigerian_Stock_Exchange_logo.png/600px-Nigerian_Stock_Exchange_logo.png", width=100)
    st.header("📊 Configuration")
    
    organizations = df_all['Organisation'].unique()
    default_idx = int(np.where(organizations == 'NSE')[0][0]) if 'NSE' in organizations else 0
    selected_org = st.selectbox("Select Ticker / Organization:", organizations, index=default_idx)

    df_org = df_all[df_all['Organisation'] == selected_org].sort_values('Date').copy()
    last_available_date = df_org['Date'].max()

    st.markdown("### 📅 Forecast Horizon")
    
    min_date = last_available_date + pd.Timedelta(days=1)
    default_date = max(min_date.date(), datetime.now().date())

    prediction_date = st.date_input(
        "Target Date:",
        value=default_date,
        min_value=min_date.date(),
        max_value=min_date.date() + timedelta(days=730)
    )
    prediction_date = pd.to_datetime(prediction_date)

    st.markdown("---")
    model_name = getattr(model, 'name', 'RNN Model')
    st.success(f"**Engine:** ' LSTM '\n\n**Logic:** Local Scaling")

# --- 5. RECURSIVE LOGIC ---

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
            
            baseline = predictions[-1] if predictions else last_price
            min_p = baseline * (1 - MAX_DAILY_CHANGE)
            max_p = baseline * (1 + MAX_DAILY_CHANGE)
            pred_constrained = np.clip(pred_unscaled, min_p, max_p)
            if pred_constrained <= 0: pred_constrained = 0.01
            
            predictions.append(pred_constrained)
            dates.append(curr_date)
            
            next_step_unscaled = np.full((1, len(TRAINING_FEATURES)), pred_constrained)
            next_step_unscaled[0, TRAINING_FEATURES.index('Change %')] = 0.0
            next_step_scaled = local_scaler.transform(next_step_unscaled)
            current_seq = np.vstack([current_seq[1:], next_step_scaled])
            
        curr_date += timedelta(days=1)
        
    if not predictions: return None, None
    
    df_pred = pd.DataFrame({'Date': dates, 'Price': predictions})
    return predictions[-1], df_pred

# --- 6. DASHBOARD BODY ---

# Market Overview Section
st.markdown(f"### 🏢 Market Snapshot: {selected_org}")

latest_close = df_org['Price'].iloc[-1]
latest_date_str = last_available_date.strftime('%d %b, %Y')
avg_vol = df_org['Vol.'].mean()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Last Close Price", value=f"₦{latest_close:,.2f}", delta=f"As of {latest_date_str}", delta_color="off")

with col2:
    st.metric(label="Average Volume", value=f"{avg_vol/1e6:.2f}M")

with col3:
    total_data_points = len(df_org)
    st.metric(label="Historical Data Points", value=f"{total_data_points}")

st.markdown("<br>", unsafe_allow_html=True) # Spacer

# Prediction Button Area
if st.button("🚀 Run Prediction Analysis", type="primary"):
    with st.spinner(f"Processing Market Data for {selected_org}..."):
        final_price, df_path = predict_recursive(df_org, model, prediction_date)
        
    if final_price is not None:
        # Calculations
        price_change = final_price - latest_close
        percent_change = (price_change / latest_close) * 100
        
        st.markdown("---")
        st.subheader("🔮 Forecast Results")
        
        # Result Cards
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            st.metric(
                label=f"Projected Price ({prediction_date.strftime('%d %b, %Y')})", 
                value=f"₦{final_price:,.2f}", 
                delta=f"{price_change:+,.2f} ({percent_change:+.2f}%)"
            )
        
        with res_col2:
            trend_text = "BULLISH (UP)" if percent_change > 0 else "BEARISH (DOWN)"
            trend_color = "#006400" if percent_change > 0 else "#8B0000"
            icon = "📈" if percent_change > 0 else "📉"
            
            st.markdown(f"""
            <div class="trend-card">
                <div style="font-size: 2.5rem;">{icon}</div>
                <div style="font-weight: bold; color: {trend_color}; font-size: 1.2rem; margin-top: 5px;">{trend_text}</div>
                <div style="color: #666; font-size: 0.9rem;">Predicted Trend</div>
            </div>
            """, unsafe_allow_html=True)

        # Plotting Section
        st.markdown("### 💹 Price Trajectory Analysis")
        
        # Create a container for the chart with white background
        with st.container():
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot Data (Last 365 Days + Forecast)
            history_plot = df_org.tail(365)
            
            # Historical Line
            ax.plot(history_plot['Date'], history_plot['Price'], label='Historical (1 Year)', color='#6c757d', linewidth=1.5, alpha=0.8)
            # Forecast Line
            ax.plot(df_path['Date'], df_path['Price'], label='AI Forecast', color='#007A33', linewidth=2.5)
            
            # End Point Marker
            ax.scatter(df_path['Date'].iloc[-1], final_price, color='#dc3545', s=120, zorder=5, edgecolors='white', linewidth=2, label='Target Date')
            
            # Chart Styling
            ax.set_title(f"{selected_org}: Historical vs Predicted Trend", fontsize=14, fontweight='bold', pad=20, color='#333')
            ax.set_ylabel("Price (₦)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Timeline", fontsize=12, fontweight='bold')
            
            # Grid and Spines
            ax.grid(True, which='major', linestyle='--', alpha=0.3, color='#bbb')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            
            ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
            
            # Add chart to Streamlit
            st.pyplot(fig)
        
        # Data Table
        with st.expander("📄 View Detailed Forecast Data"):
            st.dataframe(df_path.style.format({"Price": "₦{:,.2f}"}), use_container_width=True)
            
    else:
        st.error("Insufficient historical data to generate a forecast.")
else:
    st.info("Select a target date and click 'Run Prediction Analysis' to see the future.")