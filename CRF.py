import streamlit as st
import numpy as np
import pandas as pd
import base64
import time

from datetime import datetime
from cryptocmd import CmcScraper
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 
from plotly import graph_objs as go
from PIL import Image


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
    return df

### LOAD THE DATA
df = load_data(selected_ticker)

### Initialise scraper 
scraper = CmcScraper(selected_ticker)

##############################################################################################


data = scraper.get_dataframe()
data_copy = scraper.get_dataframe()
data['Date'] = pd.to_datetime(data['Date']).dt.date


st.subheader(f'Historical data of {selected_ticker}') #display
st.write(data.head(5)) # display data frame


###############################################################################################

#DISPLAY RAW DATA TABLE
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True, xaxis_title="Dates",yaxis_title="Closed Data Prices")
	st.plotly_chart(fig)

def plot_raw_data_log():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
	fig.update_yaxes(type="log")
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True, xaxis_title="Dates",yaxis_title="Closed Data Prices")
	st.plotly_chart(fig)


plot_log = st.checkbox("Plot log scale")
if plot_log:
	plot_raw_data_log()
else:
	plot_raw_data()




data_copy['month'] = data_copy['Date'].apply(lambda x: x.month)
data_copy['year'] = data_copy['Date'].apply(lambda x: x.year)
data_copy['day'] = data_copy['Date'].apply(lambda x: x.day)
data_copy['week'] = data_copy['Date'].apply(lambda x: x.week)


monthly=data_copy.groupby('month').agg('mean')
monthly.reset_index(inplace=True)
monthly = monthly.rename(columns = {'index':'month'})


yearly=data_copy.groupby('year').agg('mean')
yearly.reset_index(inplace=True)
yearly = yearly.rename(columns = {'index':'year'})

daily=data_copy.groupby('day').agg('mean')
daily.reset_index(inplace=True)
daily = daily.rename(columns = {'index':'day'})
    
weekly=data_copy.groupby('week').agg('mean')
weekly.reset_index(inplace=True)
weekly = weekly.rename(columns = {'index':'week'})
    
    

    ######## Daily ######## 

fig4 = go.Figure()
fig4.add_trace(go.Scatter(y=daily['Close'], x=daily['day']))
fig4.layout.update(title_text='Daily Data', xaxis_rangeslider_visible=True, xaxis_title="Days",yaxis_title="Daily Data Prices")
st.plotly_chart(fig4)
    
  ######## Weekly ######## 
  

fig5 = go.Figure()
fig5.add_trace(go.Scatter(y=weekly['Close'], x=weekly['week']))
fig5.layout.update(title_text='Weekly Data', xaxis_rangeslider_visible=True, xaxis_title="Weeks",yaxis_title="Weekly Data Prices")
st.plotly_chart(fig5)
    
  ######## Monthly ######## 
  
 
fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=monthly['Close'], x=monthly['month']))
fig2.layout.update(title_text='Monthly Data', xaxis_rangeslider_visible=True, xaxis_title="Months",yaxis_title="Monthly Data Prices")
st.plotly_chart(fig2)
    
    
  ######## Yearly ######## 
 

fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=yearly['Close'], x=yearly['year']))
fig3.layout.update(title_text='Yearly Data', xaxis_rangeslider_visible=True, xaxis_title="Years",yaxis_title="Yearly Data Prices")
st.plotly_chart(fig3)
    



###########################################################################################

rev_data=data.reindex(index=data.index[::-1])# reverse data from first to last- last to first

closed_prices_data=rev_data[['Close']].values.reshape(-1, 1) ##### get closed prices from data

Scale=MinMaxScaler()

Scaled_data=Scale.fit(closed_prices_data)#datascaler
X=Scaled_data.transform(closed_prices_data) #####normalizing the data
X=X.reshape(X.shape[0],)


#samples split
X_samples=list()#for predicting data
y_samples=list()#

NumRows=len(X)
prediction_days=10 #next day's Price Prediction is based on last how many past day's prices
Future_Steps=int(st.sidebar.number_input('Input how many days the application will predict:', min_value=0, max_value=365, value=1, step=1)) # predicting x days from based days

if st.button("Predict"):
    
    with st.spinner('Wait for the algorithm to finish'):
       
       
###########################################################################################

# TRAINING OF DATA
        for i in range(prediction_days,NumRows-Future_Steps,1):
    
            x_sample_data=X[i-prediction_days:i]
            y_sample_data=X[i:i+Future_Steps]
            X_samples.append(x_sample_data)
            y_samples.append(y_sample_data)

#reshape input as 3D
        X_data=np.array(X_samples)
        X_data=X_data.reshape(X_data.shape[0],X_data.shape[1],1)


#y data is a single column only
        y_data=np.array(y_samples)


# num of testing data records
        test_record=5
#split data to train and test
        X_train=X_data[:-test_record]
        X_test=X_data[-test_record:]
        y_train=y_data[:-test_record]
        y_test=y_data[-test_record:]

#define inputs for LSTM
        Steps=X_train.shape[1]
        Features=X_train.shape[2]



###################################################################################################
# LSTM MODEL
        model = Sequential()

#first hidden layer and LSTM layer
        model.add(LSTM(units=50, activation='relu', input_shape=(Steps,Features),return_sequences=True))  

#second layer
        model.add(LSTM(units=25, activation='relu', input_shape=(Steps,Features),return_sequences=True))

#third layer
        model.add(LSTM(units=25, activation='relu',return_sequences=False))

#Output layer
        model.add(Dense(units=Future_Steps))

#complile RNN
        model.compile(optimizer='adam', loss='mean_squared_error')


# measure time taken for model to train
        StartTime=time.time()

#fit the RNN to Training set
        model.fit(X_train, y_train, epochs=25, batch_size=32)
       
        EndTime=time.time()
    
        st.success('Done!')
    
    
        print("##Total time taken:" ,round((EndTime-StartTime)/60),"Minutes ##")


#############################################################################################

# PREDICT Number of days
        for i in range(prediction_days,NumRows,1):
            
            X_days=rev_data[i-prediction_days:]
        
        Last_X_Days_Prices=closed_prices_data[-prediction_days:]
 
#Days of predicted values
        Dates = pd.DataFrame(pd.date_range(datetime.today(), periods=Future_Steps).tolist(), columns=['Date'])

        Dates['Date'] = pd.to_datetime(Dates['Date']).dt.date
   
 
# Reshaping the data to (-1,1 )because its a single entry
        Last_X_Days_Prices=Last_X_Days_Prices.reshape(-1, 1)
 
# Scaling the data on the same level on which model was trained
        X_test=Scaled_data.transform(Last_X_Days_Prices)
 
        NumberofSamples=1
        TimeSteps=X_test.shape[0]
        NumberofFeatures=X_test.shape[1]

# Reshaping the data as 3D input
        X_test=X_test.reshape(NumberofSamples,TimeSteps,NumberofFeatures)
 
# Generating the predictions for next X days
        Next_XDays_Price = model.predict(X_test)
 

# Generating the prices in original scale
        Next_XDays_Price = Scaled_data.inverse_transform(Next_XDays_Price)
        Predicted_Multiple_Data = pd.DataFrame(Next_XDays_Price)
       
     
################################################################################################

    
        st.subheader(f'Plot data using {prediction_days} days from historical data')
        st.write(X_days.head(prediction_days))


        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=X_days['Close'], x=X_days['Date']))
        fig1.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True, xaxis_title="Dates",yaxis_title="Prices")
        st.plotly_chart(fig1)



        st.subheader("Forecast components")
        
        st.subheader(f'Predicted values for {Future_Steps} days')
        st.write(Predicted_Multiple_Data .head())

        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(y=Predicted_Multiple_Data .iloc[0], x=Dates['Date']))
        fig6.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True, xaxis_title="Dates",yaxis_title="Predicted Prices")
        st.plotly_chart(fig6)

   