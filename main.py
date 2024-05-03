import streamlit as st
import yfinance as yf
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import base64
import pandas as pd

# Define constants
start_date = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")
stocks = ('NFLX','TSLA', 'AMZN', 'MSFT', 'NVDA', 'CRM', 'GOOG',  'INTC', 'CSCO', 'ADBE', 'BAC', 'XOM')

# Function to load data
@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, start_date, today)
        full_name = yf.Ticker(ticker).info['longName']
        data.reset_index(inplace=True)
        
        return data,full_name
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# Function to plot raw data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text='Historical data', xaxis_rangeslider_visible=False)
    return fig

# Function to get image as base64
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_latest_prices(data):
    latest_date = data.iloc[-1]['Date']
    latest_price = data.iloc[-1]['Close']
    return latest_date, latest_price   

# Function to get predicted stock price for a specific date
def get_predicted_price(data, input_date):
    # Prepare data for forecasting
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    
    # Prophet model fitting
    m = Prophet(interval_width=0.95)
    m.fit(df_train)
    
    # Make future dataframe containing the input date
    future = pd.DataFrame({'ds': [input_date]})
    
    # Predict stock price for the input date
    forecast = m.predict(future)
    predicted_price = forecast.loc[0, 'yhat']
    
    return predicted_price

# Main function
def main():
    # Background image styling
    img = get_img_as_base64("image.avif")
    page_bg_img = f"""
            <style>
                body {{
                    background-image: url("data:image/png;base64,{img}");
                    background-size: cover;
                    background-position: top right;
                    background-repeat: no-repeat;
                    font-family: 'Times New Roman', sans-serif;
                    color: white;
                }}
            </style>
        """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title('Stock Forecasting')
    
    # Load data and display loading message
    selected_stock = st.sidebar.selectbox('Select Stock for prediction', stocks)
    data,full_name= load_data(selected_stock)
    if data is not None:
       
        st.sidebar.success('Data loaded successfully!')
        # Plot raw data
        # Input field for user to enter a date
        input_date = st.date_input(f"Enter a date to get predicted stock price for: {full_name}", value=date.today())

        # Get predicted stock price for the input date
        predicted_price = get_predicted_price(data, input_date)

        st.write(f"Predicted {selected_stock} stock price on {input_date}: ${predicted_price:.2f}")
        st.subheader(f"Historical Data Plot for: {full_name}")
        fig = plot_raw_data(data)
        Closing_date, latest_closing_price = get_latest_prices(data)
        st.plotly_chart(fig)

        # Forecasting
       
        n_years = st.sidebar.slider('Select number of years to predict:', 1, 5)
        period = n_years * 365

        # Prepare data for forecasting
        df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

        # Prophet model fitting and forecasting
        m = Prophet(interval_width=0.95)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period, include_history=True)
        forecast = m.predict(future)

        # Plot raw data and forecasted values
        st.subheader('Raw Data vs Forecast')
        fig_combined = go.Figure()
        fig_combined.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Raw Data"))
        
        # Filter forecast to start from where raw data ends
        forecast_before_end_raw_data = forecast[forecast['ds'] <= data['Date'].iloc[-1]]
        forecast_after_end_raw_data = forecast[forecast['ds'] > data['Date'].iloc[-1]]
        
        fig_combined.add_trace(go.Scatter(x=forecast_before_end_raw_data['ds'], y=forecast_before_end_raw_data['yhat'], name="Forecast", line=dict(dash='dash')))
        fig_combined.add_trace(go.Scatter(x=forecast_after_end_raw_data['ds'], y=forecast_after_end_raw_data['yhat'], name="Future Forecast", line=dict(color='red')))
        
        fig_combined.layout.update(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_combined)

        # Display forecast components
        with st.expander("Click to view Forecast Components", expanded=False):
            st.write(forecast.tail())
            fig2 = m.plot_components(forecast)
            st.write(fig2)

# Run the app
if __name__ == "__main__":
    main()
