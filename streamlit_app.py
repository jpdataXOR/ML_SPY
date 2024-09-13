import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import datetime

# Function to fetch SPY data
def fetch_spy_data():
    end_date = datetime.date.today()
    start_date = end_date - pd.DateOffset(years=5)  # Fetching data for the last 5 years or more
    spy_data = yf.download("SPY", start=start_date, end=end_date)
    spy_data.reset_index(inplace=True)
    return spy_data

# Function to prepare the dataset
def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    
    # Calculate daily percentage changes
    df['Pct Change'] = df['Close'].pct_change() * 100
    
    # Calculate features
    for i in range(1, 6):
        df[f'Day-{i} Change (%)'] = df['Pct Change'].shift(i)
    
    df['ML_5-Day Moving Average (%)'] = df[['Pct Change']].rolling(window=5).mean().shift(1)
    df['ML_Volatility (5-Day) (%)'] = df[['Pct Change']].rolling(window=5).std().shift(1)
    df['ML_Day of Week'] = df['Date'].dt.dayofweek
    
    # Calculate 5-Day Forward Change for result comparison
    df['Result_5-Day Forward Change (%)'] = df['Pct Change'].shift(-5)
    
    # Define features and target
    features = ['Day-5 Change (%)', 'Day-4 Change (%)', 'Day-3 Change (%)', 'Day-2 Change (%)', 'Day-1 Change (%)', 
                'ML_5-Day Moving Average (%)', 'ML_Volatility (5-Day) (%)', 'ML_Day of Week']
    target = 'Result_5-Day Forward Change (%)'
    
    # Remove rows with NaN values in feature or target columns
    df_clean = df.dropna(subset=features + [target])
    
    # Debugging: Print information about NaN values
    print("NaN values in each column:")
    print(df_clean.isna().sum())
    print("\nShape of cleaned dataframe:", df_clean.shape)
    
    return df_clean

# Function to train model and make predictions
def train_model(df):
    features = ['Day-5 Change (%)', 'Day-4 Change (%)', 'Day-3 Change (%)', 'Day-2 Change (%)', 'Day-1 Change (%)', 
                'ML_5-Day Moving Average (%)', 'ML_Volatility (5-Day) (%)', 'ML_Day of Week']
    target = 'Result_5-Day Forward Change (%)'
    
    X = df[features]
    y = df[target]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    # Train Neural Network
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions for the entire dataset
    df['Predicted_5-Day Forward Change (%)'] = model.predict(X_scaled)
    
    return df, model, scaler

# Main Streamlit app
def main():
    st.title("SPY Price Prediction with Neural Network")

    # Fetch data
    st.write("Fetching data...")
    try:
        spy_data = fetch_spy_data()
    except Exception as e:
        st.error(f"An error occurred during data fetching: {str(e)}")
        return

    # Prepare data
    st.write("Preparing data...")
    try:
        df = prepare_data(spy_data)
    except Exception as e:
        st.error(f"An error occurred during data preparation: {str(e)}")
        st.write("Debugging information:")
        st.write("NaN values in DataFrame:")
        st.write(df.isna().sum())
        return

    # Train model and make predictions
    st.write("Training model and making predictions...")
    try:
        df, model, scaler = train_model(df)
        
        # Display the table with all relevant information
        st.write("Data with features, actual results, and model predictions:")
        columns_to_display = ['Date', 'Close', 'Day-5 Change (%)', 'Day-4 Change (%)', 'Day-3 Change (%)', 'Day-2 Change (%)', 
                              'Day-1 Change (%)', 'ML_5-Day Moving Average (%)', 'ML_Volatility (5-Day) (%)', 
                              'ML_Day of Week', 'Result_5-Day Forward Change (%)', 'Predicted_5-Day Forward Change (%)']
        
        # Filter out columns that don't exist and sort by date (most recent first)
        display_df = df[columns_to_display].sort_values('Date', ascending=False)
        st.dataframe(display_df)
        
        # Display metrics
        actual = df['Result_5-Day Forward Change (%)'].dropna()
        predicted = df['Predicted_5-Day Forward Change (%)'].loc[actual.index]
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        st.write(f"Root Mean Squared Error of the model: {rmse:.4f}")

        # Plot predictions vs actual
        st.line_chart({
            "Actual": actual,
            "Predicted": predicted
        })

        # Feature importance (for neural networks, we'll use the absolute values of the first layer weights)
        feature_importance = np.abs(model.coefs_[0]).mean(axis=1)
        feature_importance = feature_importance / np.sum(feature_importance)
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

        st.write("Feature Importance:")
        st.bar_chart(feature_importance_df.set_index('Feature'))

    except Exception as e:
        st.error(f"An error occurred during model training or prediction: {str(e)}")
        st.write("Debugging information:")
        st.write(f"DataFrame shape: {df.shape}")
        st.write("NaN values in DataFrame:")
        st.write(df.isna().sum())

if __name__ == "__main__":
    main()
