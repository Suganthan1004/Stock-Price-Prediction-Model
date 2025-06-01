import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from newsapi import NewsApiClient
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime

NEWS_API_KEY = 'cf46460e66c44c80bb2aac1a7701e764'  # Replace with your actual API key
newsapi = NewsApiClient(api_key=NEWS_API_KEY)


# Streamlit UI
st.title("📈 Stock Trend Prediction")
user_input = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", "AAPL")


st.sidebar.subheader("Select Date Range")
start = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end = st.sidebar.date_input("End Date", datetime.date(2025, 4, 10))

# Ensure correct order
if start >= end:
    st.error("❗ End date must be after start date.")
    st.stop()

# Download data
df = yf.download(user_input, start=start, end=end)

if df.empty:
    st.error("No data found. Please enter a valid stock ticker.")
    st.stop()

# Show stats
st.subheader("📊 Data Description (From 2020)")
df_filtered = df[df.index >= "2020-01-01"]
st.write(df_filtered.describe())

# Matplotlib Plot: Closing Price
st.subheader("📈 Closing Price vs Time Chart")
fig1 = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.title("Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
st.pyplot(fig1)

# Matplotlib Plot: Moving Averages
st.subheader("📈 Closing Price with 100MA & 200MA")
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig2 = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close Price')
plt.plot(ma100, 'r', label='100 MA')
plt.plot(ma200, 'g', label='200 MA')
plt.title("Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

# Prepare training and testing data
data_training = df[:int(len(df)*0.70)]
data_testing = df[int(len(df)*0.70):]

scaler = MinMaxScaler(feature_range=(0, 1))
training_close = data_training[['Close']]
training_scaled = scaler.fit_transform(training_close)

x_train, y_train = [], []
for i in range(100, len(training_scaled)):
    x_train.append(training_scaled[i-100:i])
    y_train.append(training_scaled[i])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

# Build model
model = Sequential()
model.add(LSTM(60, activation='relu', return_sequences=True, input_shape=(100, 1)))
model.add(Dropout(0.2))
model.add(LSTM(60, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

opt = Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='mean_squared_error')
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, callbacks=[early_stop], verbose=0)

model.save('Stock_Predictor.keras')

# Prepare test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
test_close = final_df[['Close']]
input_data = scaler.transform(test_close)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Predict
y_pred = model.predict(x_test)
y_predicted = scaler.inverse_transform(y_pred)
y_original = scaler.inverse_transform(y_test)

# Evaluation metrics
mse = mean_squared_error(y_original, y_predicted)
mae = mean_absolute_error(y_original, y_predicted)
rmse = np.sqrt(mse)

# Plotly Interactive Plot: Prediction vs Original
st.subheader("📉 Prediction vs Original (Interactive)")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=y_original.flatten(), mode='lines', name='Original Price', line=dict(color='blue')))
fig3.add_trace(go.Scatter(y=y_predicted.flatten(), mode='lines', name='Predicted Price', line=dict(color='red')))
fig3.update_layout(title='Predicted vs Original Prices', xaxis_title='Time Steps', yaxis_title='Price')
st.plotly_chart(fig3, use_container_width=True)

st.write(f"📌 RMSE: `{rmse:.4f}`")
st.write(f"📌 MAE: `{mae:.4f}`")

def get_news(ticker):
    query = f"{ticker} stock OR {ticker} finance OR {ticker} earnings"
    try:
        articles = newsapi.get_everything(
            q=query,
            language='en',
            page_size=5
        )
        return articles['articles']
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

st.subheader(f"📰 Latest News on {user_input.upper()}")

news_articles = get_news(user_input)
if news_articles:
    for article in news_articles:
        st.markdown(f"**[{article['title']}]({article['url']})**")
        st.write(article['source']['name'], " - ", article['publishedAt'][:10])
        st.write(article['description'])
        st.write("---")
else:
    st.write("No news found.")
