import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import altair as alt
import base64

#from datetime import datetime
#from forex_python.converter import CurrencyRates
#from forex_python.bitcoin import BtcConverter

#from statsmodels.tsa.seasonal import seasonal_decompose

from cryptocmd import CmcScraper
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM ,Dropout
from plotly import graph_objs as go
from PIL import Image
# tf.keras.datasets


main_bg = "bg.jpg"
main_bg_ext = "jpg"


st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)



image = Image.open('logo.png')
st.image(image, width=500)



st.title('CryptoRush')
st.markdown('A Web Application that enables you to predict and forecast the future value of any cryptocurrency on a daily, weekly, and monthly basis.')


selected_ticker = st.sidebar.selectbox("Choose type of Crypto (i.e. BTC, ETH, BNB, XRP)",options=["BTC", "ETH", "BNB", "XRP"] )

# INITIALIZE SCRAPER
@st.cache
def load_data(selected_ticker):
    
    init_scraper = CmcScraper(selected_ticker)
    df = init_scraper.get_dataframe()
    #min_date = pd.to_datetime(min(df['Date']))
    #max_date = pd.to_datetime(max(df['Date']))


    return df

### LOAD THE DATA
df = load_data(selected_ticker)

### Initialise scraper without time interval
scraper = CmcScraper(selected_ticker)

##############################################################################################


data = scraper.get_dataframe()


st.subheader('Historical data') #display
st.write(data.head(5)) # display data frame

####################################################################################################################

#DISPLAY RAW DATA TABLE
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

def plot_raw_data_log():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
	fig.update_yaxes(type="log")
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)


plot_log = st.checkbox("Plot log scale")
if plot_log:
	plot_raw_data_log()
else:
	plot_raw_data()



###########################################################################################
    
scaler = MinMaxScaler(feature_range=(0, 1)) #normalization


scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1)) # get close price

######### change input day label

prediction_days = int(st.sidebar.number_input('Input days:', min_value=0, max_value=365, value=60, step=1))  #number of days to base prediction
#date=datetime.now()
#c=CurrencyRates()

#val = data['Close'].values[0]

#b = BtcConverter() # force_decimal=True to get Decimal rates
#late=b.get_latest_price('USD')

#st.sidebar.markdown("Convert USD to PHP")
#Price=st.sidebar.number_input('USD Amount:', value=1)
#st.sidebar.markdown("Converted USD Amount to PHP: ")
#st.sidebar.markdown(c.convert('USD','PHP',val))

#st.sidebar.markdown(f"Latest price of {selected_ticker} is")
#st.sidebar.markdown(late)


#future_day = 30       #Extension




###########################################################################################

# TRAINING OF DATA
if st.button("Predict"):

    

    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    
  
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)  ):   #computation for prediction
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x  , 0])


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

###################################################################################################

# LSTM MODEL
model = Sequential()



model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))  
model.add(Dropout(0.2)) 

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics='mse')


model.fit(x_train, y_train, epochs=25, batch_size=32)


##############################################################################################

# TESTING OF DATA
exp=data.reindex(index=data.index[::-1])

test_data =exp

actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)  

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values

model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

#############################################################################################

# PREDICT NEXT DAY

#need to be dataframe to plot

real_data=[model_inputs[len(model_inputs)+ 1 - prediction_days:len(model_inputs) + 1, 0]]
real_data=np.array(real_data)
real_data=np.reshape(real_data, (real_data.shape[0], real_data.shape[1] , 1))

#real_data_frame=pd.DataFrame()


prediction=model.predict(real_data)
prediction=scaler.inverse_transform(prediction)

print(prediction)


pred = pd.DataFrame(prediction_prices,columns=['predicted'])

reversing=pred.reindex(index=pred.index[::-1])
############################################################################################


#PLOT

plt.plot(actual_prices, color='red', label='Actual Prices')
plt.plot(prediction_prices, color='green', label='Predicted Prices')
#plt.plot(prediction, color='red', label='Predicted Prices')
plt.title('price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()
#print(df.head(5))



####################################################################################################################

    
st.subheader('Forecast data')
#st.write(.head())
st.write(reversing.head(5))

st.subheader(f'Forecast plot using {prediction_days} days from historical data')

fig1 = go.Figure()
fig1.add_trace(go.Scatter(y=reversing['predicted'], x=data['Date']))
fig1.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)




#st.subheader("Forecast components")

data['month'] = data['Date'].apply(lambda x: x.month)
data['year'] = data['Date'].apply(lambda x: x.year)
data['day'] = data['Date'].apply(lambda x: x.day)


data.groupby('month').agg('mean').plot()
plt.title('Monthly')
plt.xlabel('Month')
plt.legend(loc='upper right')

data.groupby('year').agg('mean').plot()
plt.title('Yearly')
plt.xlabel('Year')
plt.legend(loc='upper right')

data.groupby('day').agg('mean').plot()
plt.title('Daily')
plt.xlabel('Days')
plt.legend(loc='upper right')