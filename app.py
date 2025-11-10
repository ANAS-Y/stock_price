import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION AND LOADING ---

# Define file paths
MODEL_FILE = 'lstm_model_nse.h5'
SCALER_FILE = 'scaler_nse.joblib'
DATA_FILE = 'cleaned_nigerian_stock_data.csv'
LOOKBACK_PERIOD = 60 # Must match the training lookback period

st.set_page_config(layout="wide", page_title="Stock Price Prediction (LSTM)")

# Cache resources (model and scaler) so they only load once
@st.cache_resource
def load_all_files():
    """Loads model, scaler, and cleaned data from disk."""
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        df = pd.read_csv(DATA_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        return model, scaler, df
    except FileNotFoundError as e:
        st.error(f"FATAL ERROR: Required file not found. Please ensure {e.filename} and the other two files are in the same folder as app.py.")
        return None, None, pd.DataFrame()

model, scaler, df_all = load_all_files()

# --- 2. APP LAYOUT AND CHECKS ---

st.title("🇳🇬 NSE Stock Price Forecasting (LSTM RNN)")
st.caption("Developed by Abdulrashid Abubakar | Modibbo Adama University, Yola")

if df_all.empty or model is None or scaler is None:
    st.stop()

# Sidebar for selection
organizations = df_all['Organisation'].unique()
# Find the index of 'NSE' and explicitly cast it to a standard Python int()
if 'NSE' in organizations:
    default_index = int(np.where(organizations == 'NSE')[0][0])
else:
    default_index = 0
    
selected_org = st.sidebar.selectbox("Select Organisation/Ticker for Analysis:", organizations, index=default_index) 

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.info(f"Model trained on the **NSE All Share Index** using a **{LOOKBACK_PERIOD}-day** lookback window. It is applied to the selected stock.")

# Filter data for the selected organization
df_org = df_all[df_all['Organisation'] == selected_org].sort_values('Date').copy()


# --- 3. PREDICTION FUNCTION ---

def predict_next_day(org_data, model, scaler):
    """Generates the prediction for the next trading day."""
    
    if len(org_data) < LOOKBACK_PERIOD:
        return None, None, None # Not enough data
        
    # Use the 'Price' column for prediction
    last_prices = org_data['Price'].values[-LOOKBACK_PERIOD:].reshape(-1, 1)
    
    # 1. Scale the input data using the trained scaler
    scaled_input = scaler.transform(last_prices)
    
    # 2. Reshape for LSTM input: [1, 60, 1]
    X_test = np.reshape(scaled_input, (1, LOOKBACK_PERIOD, 1))
    
    # 3. Predict the scaled price
    scaled_prediction = model.predict(X_test, verbose=0)
    
    # 4. Inverse transform to get the actual Naira price
    predicted_price = scaler.inverse_transform(scaled_prediction)[0, 0]
    
    # Calculate the next trading date
    last_date = org_data['Date'].max()
    next_date = last_date + pd.Timedelta(days=1)
    
    # Advance to the next weekday (simple trading day approximation)
    while next_date.weekday() > 4: # 5=Saturday, 6=Sunday
        next_date += pd.Timedelta(days=1)
        
    return predicted_price, last_date, next_date

# --- 4. MAIN METRICS DISPLAY ---

if not df_org.empty:
    latest_price = df_org['Price'].iloc[-1]
    latest_date = df_org['Date'].iloc[-1].strftime('%Y-%m-%d')
    mean_volume = df_org['Vol.'].mean() / 1_000_000 # Convert to millions

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(f"Latest Close Price ({latest_date})", f"₦{latest_price:,.2f}")
    with col2:
        st.metric("Average Volume (Millions)", f"{mean_volume:,.2f}M")
    with col3:
        st.metric("Total Data Points", f"{len(df_org):,}")
else:
    st.stop()


# --- 5. PREDICTION SECTION ---
st.header(f"Forecast for {selected_org}")

if st.button(f"Predict Next Trading Day Price for {selected_org}"):
    predicted_price, last_date, next_date = predict_next_day(df_org, model, scaler)
    
    if predicted_price is not None:
        st.success(f"**Predicted Closing Price for {selected_org} on {next_date.strftime('%Y-%m-%d')}:**")
        st.balloons()
        
        # Display the main prediction
        st.markdown(f"## ₦{predicted_price:,.2f}")
        
        # Comparison to latest price
        change_pct = (predicted_price - latest_price) / latest_price * 100
        st.markdown(f"*(Change from previous close: **{change_pct:+.2f}%**)*")
        
        if change_pct > 0:
            st.markdown("**(Predicted Trend: UP)**")
        elif change_pct < 0:
            st.markdown("**(Predicted Trend: DOWN)**")
        else:
            st.markdown("**(Predicted Trend: NEUTRAL)**")
            
    else:
        st.warning(f"Not enough historical data for {selected_org} (Requires at least {LOOKBACK_PERIOD} days).")


# --- 6. VISUALIZATION ---
st.header("Historical Price Trend")

# Create a figure for plotting
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the historical closing price
ax.plot(df_org['Date'], df_org['Price'], label='Actual Closing Price', color='#007A33', linewidth=2) # NSE Green
ax.set_title(f'Historical Closing Price for {selected_org}', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (₦)', fontsize=12)
ax.legend()
ax.grid(True, linestyle=':', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)