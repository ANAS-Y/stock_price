import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os 
from copy import deepcopy # For safe data copying

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
st.sidebar.info(f"Model trained on the **NSE All Share Index** using a **{LOOKBACK_PERIOD}-day** lookback window. This model now performs **recursive multi-step forecasting** to predict the price for the target date.")


# --- 3. RECURSIVE PREDICTION FUNCTION (UPDATED) ---

def predict_recursive(org_data, model, scaler, target_date, lookback):
    """
    Generates multi-step, recursive predictions up to the target_date.
    Returns: The final predicted price and a DataFrame of the entire prediction path.
    """
    
    last_date = org_data['Date'].max()
    
    # 1. Prepare initial sequence (scaled)
    last_prices = org_data['Price'].values[-lookback:].reshape(-1, 1)
    scaled_input_sequence = scaler.transform(last_prices)
    
    # Use a deepcopy to modify the sequence iteratively
    current_input_sequence = deepcopy(scaled_input_sequence)
    
    prediction_dates = []
    prediction_prices = []
    current_date = last_date + pd.Timedelta(days=1)

    while current_date <= target_date:
        
        # Skip weekends/non-trading days (approximation)
        if current_date.weekday() < 5: # Monday is 0, Friday is 4
            
            # Reshape for LSTM: [1, lookback, 1]
            X_test = np.reshape(current_input_sequence, (1, lookback, 1))
            
            # Predict the next scaled price
            scaled_prediction = model.predict(X_test, verbose=0)
            
            # Inverse transform to get the actual Naira price
            predicted_price = scaler.inverse_transform(scaled_prediction)[0, 0]
            
            # Store prediction
            prediction_dates.append(current_date)
            prediction_prices.append(predicted_price)
            
            # Update the input sequence for the next step (recursive part)
            # 1. Remove the oldest price (the first element)
            current_input_sequence = np.delete(current_input_sequence, 0, axis=0)
            
            # 2. Append the new predicted scaled price to the end
            current_input_sequence = np.append(current_input_sequence, scaled_prediction, axis=0)

        # Move to the next day
        current_date += pd.Timedelta(days=1)
        
    if not prediction_prices:
        return None, None

    # Final result is the last predicted price
    final_predicted_price = prediction_prices[-1]
    
    # Create prediction DataFrame
    df_prediction_path = pd.DataFrame({
        'Date': prediction_dates,
        'Price': prediction_prices
    })
    
    return final_predicted_price, df_prediction_path

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

if st.button(f"Generate Multi-Step Forecast up to {prediction_date.strftime('%Y-%m-%d')}"):
    
    with st.spinner('Generating recursive forecast...'):
        predicted_price, df_prediction_path = predict_recursive(df_org, model, scaler, prediction_date, LOOKBACK_PERIOD)
    
    if predicted_price is not None:
        
        # Store prediction path in session state to use in the visualization section (Section 6)
        st.session_state['df_prediction_path'] = df_prediction_path
        st.session_state['prediction_date'] = prediction_date
        
        st.success(f"**Predicted Closing Price for {selected_org} on {prediction_date.strftime('%Y-%m-%d')}:**")
        st.balloons()
        
        # Display the main prediction
        st.markdown(f"## ₦{predicted_price:,.2f}")
        
        # Comparison to latest price
        change_pct = (predicted_price - latest_price) / latest_price * 100
        st.markdown(f"*(Total change from previous close: **{change_pct:+.2f}%**)*")
        
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


# Add the recursive prediction path to the chart if available in session state
if 'df_prediction_path' in st.session_state and not st.session_state['df_prediction_path'].empty:
    df_path = st.session_state['df_prediction_path']
    final_predicted_price = df_path['Price'].iloc[-1]
    
    # 1. Plot the entire predicted path
    ax.plot(df_path['Date'], df_path['Price'], label='Recursive Forecast', color='orange', linestyle='--', linewidth=1.5)
    
    # 2. Highlight the final prediction point
    ax.scatter(df_path['Date'].iloc[-1], final_predicted_price, color='red', marker='*', s=200, label='Final Predicted Price')
    
    ax.legend()

st.pyplot(fig)