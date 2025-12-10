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
MODEL_FILE = 'lstm_model_nse_augmented.h5' # NEW: Model trained on augmented NSE data
DATA_FILE = 'cleaned_nigerian_stock_data.csv' # Historic data for all stocks
LOOKBACK_PERIOD = 60
TRAINING_FEATURES = ['Price', 'Open', 'High', 'Low', 'Change %']
MAX_DAILY_CHANGE = 0.05 # 5% constraint

st.set_page_config(layout="wide", page_title="Stock Price Prediction (NSE Model)")

@st.cache_resource
def load_resources():
    """Loads model and data. Note: No global scaler needed for prediction."""
    
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
                
        return model, df
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

model, df_all = load_resources()

# --- 2. SIDEBAR CONFIGURATION ---

st.title("🇳🇬 NSE Stock Price Forecasting")
st.caption("Powered by Augmented NSE Data Model (23,600+ Records)")

if df_all is None or model is None:
    st.stop()

organizations = df_all['Organisation'].unique()
default_idx = int(np.where(organizations == 'NSE')[0][0]) if 'NSE' in organizations else 0
selected_org = st.sidebar.selectbox("Select Ticker:", organizations, index=default_idx)

df_org = df_all[df_all['Organisation'] == selected_org].sort_values('Date').copy()
last_available_date = df_org['Date'].max()

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Settings")

min_date = last_available_date + pd.Timedelta(days=1)
default_date = max(min_date.date(), datetime.now().date())

prediction_date = st.sidebar.date_input(
    "Target Date:",
    value=default_date,
    min_value=min_date.date(),
    max_value=min_date.date() + timedelta(days=730)
)
prediction_date = pd.to_datetime(prediction_date)

st.sidebar.info(f"Model: **NSE Augmented**\nSplit: **80% Train / 20% Test**\nStrategy: **Localized Scaling**")

# --- 3. RECURSIVE PREDICTION LOGIC ---

def predict_recursive(org_data, model, target_date):
    """Recursive prediction using localized scaling for stability."""
    
    if len(org_data) < LOOKBACK_PERIOD:
        return None, None

    # 1. LOCAL SCALING (Crucial for applying NSE model to other stocks)
    local_scaler = MinMaxScaler(feature_range=(0, 1))
    data_values = org_data[TRAINING_FEATURES].values
    local_scaler.fit(data_values) # Fit ONLY on selected stock's history
    
    # 2. Initial Sequence
    last_60 = data_values[-LOOKBACK_PERIOD:]
    current_seq = local_scaler.transform(last_60) # Scale to 0-1
    
    predictions = []
    dates = []
    curr_date = last_available_date + timedelta(days=1)
    
    PRICE_IDX = TRAINING_FEATURES.index('Price')
    last_price = data_values[-1, PRICE_IDX] # Unscaled last price

    while curr_date <= target_date:
        if curr_date.weekday() < 5: # Trading days only
            # Reshape for LSTM (1, 60, 5)
            input_tensor = current_seq.reshape(1, LOOKBACK_PERIOD, len(TRAINING_FEATURES))
            
            # Predict (Returns scaled price 0-1)
            pred_scaled = model.predict(input_tensor, verbose=0)[0, 0]
            
            # Inverse Transform to get Naira
            dummy = np.zeros((1, len(TRAINING_FEATURES)))
            dummy[0, PRICE_IDX] = pred_scaled
            pred_unscaled = local_scaler.inverse_transform(dummy)[0, PRICE_IDX]
            
            # --- STABILITY CONSTRAINT ---
            baseline = predictions[-1] if predictions else last_price
            min_p = baseline * (1 - MAX_DAILY_CHANGE)
            max_p = baseline * (1 + MAX_DAILY_CHANGE)
            pred_constrained = np.clip(pred_unscaled, min_p, max_p)
            if pred_constrained <= 0: pred_constrained = 0.01
            
            # Store
            predictions.append(pred_constrained)
            dates.append(curr_date)
            
            # Update Sequence
            # We assume predicted price applies to all price features for the next step (stability)
            next_step_unscaled = np.full((1, len(TRAINING_FEATURES)), pred_constrained)
            # Important: Keep Change % as 0.0 (neutral assumption)
            next_step_unscaled[0, TRAINING_FEATURES.index('Change %')] = 0.0
            
            next_step_scaled = local_scaler.transform(next_step_unscaled)
            
            # Shift sequence
            current_seq = np.vstack([current_seq[1:], next_step_scaled])
            
        curr_date += timedelta(days=1)
        
    if not predictions: return None, None
    
    df_pred = pd.DataFrame({'Date': dates, 'Price': predictions})
    return predictions[-1], df_pred

# --- 4. DISPLAY ---

col1, col2 = st.columns(2)
with col1:
    st.metric("Latest Close", f"₦{df_org['Price'].iloc[-1]:,.2f}")
    st.text(f"Date: {last_available_date.strftime('%Y-%m-%d')}")
with col2:
    st.metric("Volume (Avg)", f"{df_org['Vol.'].mean()/1e6:.2f}M")

st.header(f"Forecast for {selected_org}")

if st.button("Generate Forecast"):
    with st.spinner("Calculating..."):
        final_price, df_path = predict_recursive(df_org, model, prediction_date)
        
    if final_price is not None:
        st.success(f"Forecast for {prediction_date.strftime('%Y-%m-%d')}: **₦{final_price:,.2f}**")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_org['Date'], df_org['Price'], label='Historical', color='gray')
        ax.plot(df_path['Date'], df_path['Price'], label='Forecast', color='green', linestyle='--')
        ax.scatter(df_path['Date'].iloc[-1], final_price, color='red', s=100)
        ax.set_title(f"{selected_org} Price Projection")
        ax.set_ylabel("Price (₦)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Data table
        with st.expander("View Forecast Data"):
            st.dataframe(df_path)
            
    else:
        st.error("Insufficient data for forecasting.")