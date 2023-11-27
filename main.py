import streamlit as st
from datetime import datetime

import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AApl", "GOOG", "MSFT", "GME", "AMC", "TSLA", "BTC-USD", "ETH-USD")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

slider = st.slider("Years of prediction:", 1, 4)
period = slider * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()










































import pandas as pd 


df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = ARIMA(df_train.y, order=(1, 1, 1))
model_fit = model.fit()
future = model_fit.predict(len(df_train), len(df_train) + period, typ="levels")

# Create a new DataFrame for 'future'
future_df = pd.DataFrame(future, columns=["y"])
future_df["ds"] = pd.date_range(start=df_train.ds.iloc[-1], periods=len(future_df) + 1, inclusive='right')


# Concatenate 'df_train' and 'future_df'
forecast = pd.concat([df_train, future_df], ignore_index=True)

st.subheader("Forecast data")
st.write(forecast.tail()) 


future_df['y'] = future_df['y'].astype(float)








import plotly.graph_objs as go

# Convert 'ds' to datetime in both DataFrames
df_train['ds'] = pd.to_datetime(df_train['ds'])
future_df['ds'] = pd.to_datetime(future_df['ds'])

# Merge 'df_train' and 'future_df'
aligned_df = pd.merge(df_train, future_df, on='ds', how='outer')









import plotly.graph_objs as go

# Plotting the observed and forecasted data
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='Observed', line=dict(color='sandybrown')))
fig1.add_trace(go.Scatter(x=future_df['ds'], y=future_df['y'], mode='lines', name='Forecast', line=dict(color='royalblue')))
fig1.update_layout(title='Forecast plot for next years', xaxis_title='Date', yaxis_title='Close Price')
st.plotly_chart(fig1)

# Plotting the forecast components
# As ARIMA does not provide a function to plot the forecast components like the Facebook Prophet model, 
# you can examine the residuals of your ARIMA model using the `plot_diagnostics()` function.
fig2 = model_fit.plot_diagnostics(figsize=(10, 8))
st.pyplot(fig2)
















# plot_raw_data()

# # Predict forecast with Prophet.
# df_train = data[['Date','Close']]
# df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# m = Prophet()
# m.fit(df_train)
# future = m.make_future_dataframe(periods=period)
# forecast = m.predict(future)

# # Show and plot forecast
# st.subheader('Forecast data')
# st.write(forecast.tail())
    
# st.write(f'Forecast plot for {n_years} years')
# fig1 = plot_plotly(m, forecast)
# st.plotly_chart(fig1)

# st.write("Forecast components")
# fig2 = m.plot_components(forecast)
# st.write(fig2)



# second option for prediction


# import pandas as pd

# # # Predict forecast with ARIMA.
# df_train = data[["Date", "Close"]]
# df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
# model = ARIMA(df_train.y, order=(1, 1, 1))
# model_fit = model.fit()
# future = model_fit.predict(len(df_train), len(df_train) + period, typ="levels")

# from pandas.tseries.offsets import DateOffset

# # Get the last date from df_train
# last_date = df_train['ds'].values[-1]

# # Generate future dates
# future_dates = [last_date + DateOffset(days=x) for x in range(1, period+1)]
# future_dates_df = pd.DataFrame(future_dates, columns=['ds'])

# # Make predictions
# future = model_fit.predict(start=len(df_train), end=len(df_train)+period-1, typ='levels')

# # Convert future to a DataFrame
# future = pd.DataFrame({'ds': future_dates_df['ds'], 'yhat': future.values})

# # Concatenate the original data with the predictions
# df_forecast = pd.concat([df_train, future], axis=0)

# # Plot the original data and the predictions
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df_forecast['ds'], y=df_forecast['y'], name='Original'))
# fig.add_trace(go.Scatter(x=df_forecast['ds'], y=df_forecast['yhat'], name='Forecast'))
# fig.layout.update(title_text='Time Series Forecast', xaxis_rangeslider_visible=True)
# st.plotly_chart(fig)




# import itertools
# import numpy as np
# from statsmodels.tsa.arima_model import ARIMA
# from sklearn.metrics import mean_squared_error

# # Define the p, d and q parameters to take any value between 0 and 2
# p = d = q = range(0, 2)

# # Generate all different combinations of p, d and q triplets
# pdq = list(itertools.product(p, d, q))

# # Initialize best_score and best_cfg
# best_score, best_cfg = float("inf"), None

# # Grid search
# for param in pdq:
#     try:
#         model = ARIMA(df_train.y, order=param)
#         model_fit = model.fit()
#         mse = mean_squared_error(df_train.y, model_fit.fittedvalues)
#         if mse < best_score:
#             best_score, best_cfg = mse, param
#         print('ARIMA%s MSE=%.3f' % (param, mse))
#     except:
#         continue

# print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))




