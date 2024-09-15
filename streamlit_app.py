import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import datetime
import requests

# Function to get list of tickers
@st.cache_data
def get_ticker_list():
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
    response = requests.get(url)
    tickers = response.text.split('\n')
    tickers = [ticker.strip() for ticker in tickers if ticker.strip()]
    
    # Add additional indices
    additional_indices = ['^SPX', '^AXJO', '^VIX', '^NDX','SPY']
    tickers = additional_indices + tickers
    
    return list(dict.fromkeys(tickers))  # Remove duplicates while preserving order

# Function to fetch instrument data
def fetch_instrument_data(symbol):
    end_date = datetime.date.today()
    start_date = end_date - pd.DateOffset(years=5)  # Fetching data for the last 5 years or more
    instrument_data = yf.download(symbol, start=start_date, end=end_date)
    instrument_data.reset_index(inplace=True)
    return instrument_data

# Function to calculate trading days left in the month/year
def calculate_trading_days_left(date, end_of_period):
    remaining_dates = pd.date_range(date, end_of_period, freq='B')  # 'B' gives business days (weekdays)
    return len(remaining_dates)

# Function to calculate end of year returns
def calculate_extrapolated_returns(df):
    df['Trading Days Left in Month'] = df['Date'].apply(lambda x: calculate_trading_days_left(x, pd.Timestamp(x.year, x.month, 1) + pd.DateOffset(months=1, days=-1)))
    df['Trading Days Left in Year'] = df['Date'].apply(lambda x: calculate_trading_days_left(x, pd.Timestamp(x.year, 12, 31)))
    
    # Calculate the average daily return over the past 10 and 5 days
    day_cols_10 = [f'Day-{i} Change (%)' for i in range(1, 11)]
    day_cols_5 = [f'Day-{i} Change (%)' for i in range(1, 6)]
    df['10-Day Avg Return (%)'] = df[day_cols_10].mean(axis=1)
    df['5-Day Avg Return (%)'] = df[day_cols_5].mean(axis=1)
    
    # Extrapolate the return till the end of the month and end of the year
    df['Extrapolated Return Till End of Month (%)'] = df['10-Day Avg Return (%)'] * df['Trading Days Left in Month']
    df['Extrapolated 10-Day Return Till End of Year (%)'] = df['10-Day Avg Return (%)'] * df['Trading Days Left in Year']
    df['Extrapolated 5-Day Return Till End of Year (%)'] = df['5-Day Avg Return (%)'] * df['Trading Days Left in Year']
    
    return df

# Function to prepare the dataset
def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    
    # Calculate daily percentage changes
    df['Pct Change'] = df['Close'].pct_change() * 100
    
    # Calculate features (Day-10 to Day-1)
    for i in range(1, 11):
        df[f'Day-{i} Change (%)'] = df['Pct Change'].shift(i)
    
    df['ML_5-Day Moving Average (%)'] = df[['Pct Change']].rolling(window=5).mean().shift(1)
    df['ML_Volatility (5-Day) (%)'] = df[['Pct Change']].rolling(window=5).std().shift(1)
    df['ML_Day of Week'] = df['Date'].dt.dayofweek
    
    # Calculate forward changes for result comparison
    df['Result_1-Day Forward Change (%)'] = df['Pct Change'].shift(-1)
    df['Result_5-Day Forward Change (%)'] = df['Pct Change'].shift(-5)
    
    # Add extrapolated return till end of month and year
    df = calculate_extrapolated_returns(df)
    
    # Define features and target for both 1-day and 5-day predictions
    features = [f'Day-{i} Change (%)' for i in range(1, 11)] + [
                'ML_5-Day Moving Average (%)', 'ML_Volatility (5-Day) (%)', 'ML_Day of Week',
                'Extrapolated Return Till End of Month (%)', 'Extrapolated 10-Day Return Till End of Year (%)',
                'Extrapolated 5-Day Return Till End of Year (%)']
    
    # Remove rows with NaN values in feature columns (but keep NaNs in the target columns)
    df_clean = df.dropna(subset=features)
    
    return df_clean, features

# Function to train model and make predictions for both 1-day and 5-day forward
def train_model(df, features):
    target_1_day = 'Result_1-Day Forward Change (%)'
    target_5_day = 'Result_5-Day Forward Change (%)'
    
    X = df[features]
    y_1_day = df[target_1_day]
    y_5_day = df[target_5_day]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split (using same split for both targets)
    X_train, X_test, y_train_1_day, y_test_1_day = train_test_split(X_scaled, y_1_day, test_size=0.2, shuffle=False)
    _, _, y_train_5_day, y_test_5_day = train_test_split(X_scaled, y_5_day, test_size=0.2, shuffle=False)
    
    # Train Neural Networks
    model_1_day = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    model_5_day = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    
    model_1_day.fit(X_train, y_train_1_day)
    model_5_day.fit(X_train, y_train_5_day)
    
    # Make predictions for the entire dataset
    df['Predicted_1-Day Forward Change (%)'] = model_1_day.predict(X_scaled)
    df['Predicted_5-Day Forward Change (%)'] = model_5_day.predict(X_scaled)
    
    return df, model_1_day, model_5_day, scaler

# CSS to make the table wider and control column width
def local_css(css_code):
    st.markdown(f'<style>{css_code}</style>', unsafe_allow_html=True)

def main():
    st.title("Stock Price Prediction with Neural Network")

    # Apply custom CSS for table styling
    local_css("""
        .streamlit-table {
            width: 100% !important;
        }
        th, td {
            padding: 5px !important;
        }
    """)

    # Get list of tickers
    tickers = get_ticker_list()

    # Add instrument selection with autocomplete
    default_instrument = "^SPX"
    instrument = st.selectbox("Select or type to search for a Yahoo Finance instrument symbol", 
                              options=tickers, 
                              index=tickers.index(default_instrument))

    if st.button("Run Prediction"):
        # Fetch data
        st.write(f"Fetching data for {instrument}...")
        try:
            instrument_data = fetch_instrument_data(instrument)
        except Exception as e:
            st.error(f"An error occurred during data fetching: {str(e)}")
            return

        # Prepare data
        st.write("Preparing data...")
        try:
            df, features = prepare_data(instrument_data)
        except Exception as e:
            st.error(f"An error occurred during data preparation: {str(e)}")
            st.write("Debugging information:")
            st.write("NaN values in DataFrame:")
            st.write(df.isna().sum())
            return

        # Train model and make predictions
        st.write("Training model and making predictions...")
        try:
            df, model_1_day, model_5_day, scaler = train_model(df, features)
            
            # Rearrange columns for display
            columns_to_display = ['Date', 'Close', 
                                  'Result_1-Day Forward Change (%)', 'Predicted_1-Day Forward Change (%)', 
                                  'Result_5-Day Forward Change (%)', 'Predicted_5-Day Forward Change (%)',
                                  'Day-10 Change (%)', 'Day-9 Change (%)', 'Day-8 Change (%)',
                                  'Day-7 Change (%)', 'Day-6 Change (%)', 'Day-5 Change (%)', 'Day-4 Change (%)',
                                  'Day-3 Change (%)', 'Day-2 Change (%)', 'Day-1 Change (%)',
                                  'ML_5-Day Moving Average (%)', 'ML_Volatility (5-Day) (%)', 'ML_Day of Week',
                                  'Extrapolated Return Till End of Month (%)', 'Extrapolated 10-Day Return Till End of Year (%)',
                                  'Extrapolated 5-Day Return Till End of Year (%)']
            
            # Filter out columns that don't exist and sort by date (most recent first)
            display_df = df[columns_to_display].sort_values('Date', ascending=False)
            st.dataframe(display_df)
            
            # Display metrics for 1-day forward predictions
            actual_1_day = df['Result_1-Day Forward Change (%)'].dropna()
            predicted_1_day = df['Predicted_1-Day Forward Change (%)'].loc[actual_1_day.index]
            mse_1_day = mean_squared_error(actual_1_day, predicted_1_day)
            rmse_1_day = np.sqrt(mse_1_day)
            st.write(f"Root Mean Squared Error of the 1-day forward prediction: {rmse_1_day:.4f}")
            
            # Display metrics for 5-day forward predictions
            actual_5_day = df['Result_5-Day Forward Change (%)'].dropna()
            predicted_5_day = df['Predicted_5-Day Forward Change (%)'].loc[actual_5_day.index]
            mse_5_day = mean_squared_error(actual_5_day, predicted_5_day)
            rmse_5_day = np.sqrt(mse_5_day)
            st.write(f"Root Mean Squared Error of the 5-day forward prediction: {rmse_5_day:.4f}")

            # Plot predictions vs actual for both 1-day and 5-day forward
            st.line_chart({
                "Actual 1-Day": actual_1_day,
                "Predicted 1-Day": predicted_1_day,
                "Actual 5-Day": actual_5_day,
                "Predicted 5-Day": predicted_5_day
            })

        except Exception as e:
            st.error(f"An error occurred during model training or prediction: {str(e)}")
            st.write("Debugging information:")
            st.write(f"DataFrame shape: {df.shape}")
            st.write("NaN values in DataFrame:")
            st.write(df.isna().sum())

if __name__ == "__main__":
    main()