import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os # Import the os library for file checking

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
    
    required_files = [MODEL_FILE, SCALER_FILE, DATA_FILE]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        st.error(
            f"**FATAL ERROR: Missing Deployment Files.**"
            f"\nPlease ensure the following files are uploaded to your Streamlit Cloud repository (in the same directory as app.py):"
            f"\n- **{', '.join(missing_files)}**"
        )
        return None, None, pd.DataFrame()
        
    try:
        # Load files now that existence is confirmed
        model = load_model(MODEL_FILE, compile=False) 
        scaler = joblib.load(SCALER_FILE)
        df = pd.read_csv(DATA_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        return model, scaler, df
        
    except Exception as e:
        st.error(f"**FATAL ERROR:** An unexpected error occurred during file loading: {e}")
        return None, None, pd.DataFrame()

model, scaler, df_all = load_all_files()

# --- 2. APP LAYOUT, CHECKS, AND DATE SELECTION ---

st.title("🇳🇬 NSE Stock Price Forecasting (LSTM RNN)")
st.caption("Developed by Abdulrashid Abubakar | Modibbo Adama University, Yola")

if df_all.empty or model is None or scaler is None:
    # If file loading failed, stop the application here
    st.stop()

# --- SIDEBAR: ORGANIZATION SELECTION ---
organizations = df_all['Organisation'].unique()
if 'NSE' in organizations:
    default_index = int(np.where(organizations == 'NSE')[0][0])
else:
    default_index = 0
    
selected_org = st.sidebar.selectbox("Select Organisation/Ticker for Analysis:", organizations, index=default_index) 

# Filter data for the selected organization
df_org = df_all[df_all['Organisation'] == selected_org].sort_values('Date').copy()
last_available_date = df_org['Date'].max()

# --- SIDEBAR: DATE SELECTION ---
st.sidebar.markdown("---")
st.sidebar.subheader("Set Prediction Date")

# Calculate the first valid prediction date (the day after the last available data)
min_date = last_available_date + pd.Timedelta(days=1)
# Default to today's date if it's after the min_date, otherwise use min_date
today = datetime.now().date()
default_date = max(min_date.date(), today)


prediction_date = st.sidebar.date_input(
    "Select Target Prediction Date (after last close):",
    value=default_date,
    min_value=min_date.date(),
    max_value=min_date.date() + timedelta(days=365*2), # Limit to 2 years ahead
    key="prediction_date_input"
)
prediction_date = pd.to_datetime(prediction_date) # Convert to datetime for comparison

# --- SIDEBAR: MODEL INFO ---
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.info(f"Model trained on the **NSE All Share Index** using a **{LOOKBACK_PERIOD}-day** lookback window. It is applied to the selected stock.")


# --- 3. PREDICTION FUNCTION ---

def predict_price(org_data, model, scaler, target_date):
    """Generates the prediction for the given target date."""
    
    if len(org_data) < LOOKBACK_PERIOD:
        return None # Not enough data
        
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
    
    return predicted_price

# --- 4. MAIN METRICS DISPLAY ---

if not df_org.empty:
    latest_price = df_org['Price'].iloc[-1]
    latest_date_str = last_available_date.strftime('%Y-%m-%d')
    mean_volume = df_org['Vol.'].mean() / 1_000_000 # Convert to millions

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(f"Latest Close Price ({latest_date_str})", f"₦{latest_price:,.2f}")
    with col2:
        st.metric("Average Volume (Millions)", f"{mean_volume:,.2f}M")
    with col3:
        st.metric("Total Data Points", f"{len(df_org):,}")
else:
    st.stop()


# --- 5. PREDICTION SECTION ---
st.header(f"Forecast for {selected_org}")

if st.button(f"Generate Forecast for {prediction_date.strftime('%Y-%m-%d')}"):
    
    # NOTE: The current model predicts only the NEXT single step. 
    # For any date other than the day after last_available_date, this is an extrapolated one-step prediction.
    predicted_price = predict_price(df_org, model, scaler, prediction_date)
    
    if predicted_price is not None:
        st.success(f"**Predicted Closing Price for {selected_org} on {prediction_date.strftime('%Y-%m-%d')}:**")
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

# Add the prediction point to the chart if a prediction was just made
if 'predicted_price' in locals() and predicted_price is not None:
    ax.scatter(prediction_date, predicted_price, color='red', marker='*', s=200, label='Predicted Price')
    ax.legend()

st.pyplot(fig)