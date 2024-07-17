import pandas as pd
import datetime as dt
import urllib.request, json
import os
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import time
from datetime import date
import pickle
import get_data

def def_model():
    np.random.seed(42)
    data = get_data.fetch_alu_mcx()
    dates = data.index.values
    data2 = get_data.fetch_alu_mcx()
    max_date= (data2.index.max() + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    FullData=data[['Price']].values
    sc = StandardScaler()
    DataScaler = sc.fit(FullData)
    X=DataScaler.transform(FullData)

    X_samples = list()
    y_samples = list()
    dates_samples = []

    NumerOfRows = len(X)
    TimeSteps=10  # next day's Price Prediction is based on last how many past day's prices
    FutureTimeSteps=5 
    for i in range(TimeSteps , NumerOfRows-FutureTimeSteps , 1):
        x_sample = X[i-TimeSteps:i]
        y_sample = X[i:i+FutureTimeSteps]
        date_sample = dates[i + FutureTimeSteps - 1] 
        X_samples.append(x_sample)
        y_samples.append(y_sample)
        dates_samples.append(date_sample)

    X_data=np.array(X_samples)
    X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
    y_data=np.array(y_samples)
    dates_data = np.array(dates_samples)

    TestingRecords=5
    X_train=X_data[:-TestingRecords]
    X_test=X_data[-TestingRecords:]
    y_train=y_data[:-TestingRecords]
    y_test=y_data[-TestingRecords:]
    dates_test = dates_data[-TestingRecords:]
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # print('Dates for X_test:', dates_test)

    TimeSteps=X_train.shape[1]
    TotalFeatures=X_train.shape[2]
    # print("Number of TimeSteps:", TimeSteps)
    # print("Number of Features:", TotalFeatures)

    os.chdir(fr"{os.getenv('BASE_DIR')}")
    filename = 'finalized_model5.sav'
    regressor = pickle.load(open(filename, 'rb'))

    dates_test = pd.date_range(max_date, periods=5) #date.today()
    predicted_Price = regressor.predict(X_test)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)
    predicted_df = pd.DataFrame({'Date': dates_test, 'Predicted_Price': predicted_Price[:, 0]})
    # print(predicted_df)
    # print((predicted_df.shape))
    # for x in range(5):
    #     print(predicted_df.iloc[x]['Date'])

    predicted_Price = regressor.predict(X_test)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)

    #y_test.shape
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
    return predicted_df


def model_train():
    days=[5,15,30]
    for day1 in days:
        if(day1==5):
            day2=int(10)
        elif(day1==15):
            day2=int(30)
        elif(day1==30):
            day2=int(60)
        np.random.seed(42)
        #tf.random.set_seed(42)
        data = get_data.fetch_alu_mcx()
        #data = data.iloc[::-1]
        #data.index = pd.to_datetime(data.index)
        dates = data.index.values

        FullData=data[['Price']].values
        sc = StandardScaler()
        DataScaler = sc.fit(FullData)
        X=DataScaler.transform(FullData)

        X_samples = list()
        y_samples = list()
        dates_samples = []

        NumerOfRows = len(X)
        TimeSteps=day2  # next day's Price Prediction is based on last how many past day's prices
        FutureTimeSteps=day1 
        for i in range(TimeSteps , NumerOfRows-FutureTimeSteps , 1):
            x_sample = X[i-TimeSteps:i]
            y_sample = X[i:i+FutureTimeSteps]
            date_sample = dates[i + FutureTimeSteps - 1] 
            X_samples.append(x_sample)
            y_samples.append(y_sample)
            dates_samples.append(date_sample)

        X_data=np.array(X_samples)
        X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
        y_data=np.array(y_samples)
        dates_data = np.array(dates_samples)

        TestingRecords=day1
        X_train=X_data[:-TestingRecords]
        X_test=X_data[-TestingRecords:]
        y_train=y_data[:-TestingRecords]
        y_test=y_data[-TestingRecords:]
        dates_test = dates_data[-TestingRecords:]
        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_test.shape)
        # print(y_test.shape)
        # print('Dates for X_test:', dates_test)

        TimeSteps=X_train.shape[1]
        TotalFeatures=X_train.shape[2]
        # print("Number of TimeSteps:", TimeSteps)
        # print("Number of Features:", TotalFeatures)

        regressor = Sequential() #relu
        regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
        regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
        regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))
        regressor.add(Dense(units = FutureTimeSteps))
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        StartTime=time.time()
        regressor.fit(X_train, y_train, batch_size = 30, epochs = 100,verbose=1)
        EndTime=time.time()
        print("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')

        dates_test = pd.date_range('2024-06-20', periods=day1) #date.today()
        predicted_Price = regressor.predict(X_test)
        predicted_Price = DataScaler.inverse_transform(predicted_Price)
        predicted_df = pd.DataFrame({'Date': dates_test, 'Predicted_Price': predicted_Price[:, 0]})
        # print(predicted_df)

        predicted_Price = regressor.predict(X_test)
        predicted_Price = DataScaler.inverse_transform(predicted_Price)

        #y_test.shape
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

        true_values = DataScaler.inverse_transform(y_test)

        mae = mean_absolute_error(true_values, predicted_Price)
        mse = mean_squared_error(true_values, predicted_Price)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_values, predicted_Price)
        print('Mean Absolute Error:', mae)
        print('Mean Squared Error:', mse)
        print('Root Mean Squared Error:', rmse)
        print('R-squared:', r2)
        os.chdir(fr"{os.getenv('BASE_DIR')}")
        # save the model to disk
        filename = 'finalized_model'+str(day1)+'.sav'
        pickle.dump(regressor, open(filename, 'wb'))

        np.mean(np.abs((true_values - predicted_Price) / true_values)) * 100

def get_prediction(day1):
    np.random.seed(42)
    data = get_data.fetch_alu_mcx()
    # print(data.head())
    #data = data.iloc[::-1]
    #data.index = pd.to_datetime(data.index)
    dates = data.index.values

    data2 = get_data.fetch_alu_mcx()
    max_date= (data2.index.max() + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    if(day1==5):
        day2=int(10)
    elif(day1==15):
        day2=int(30)
    elif(day1==30):
        day2=int(60)

    FullData=data[['Price']].values
    sc = StandardScaler()
    DataScaler = sc.fit(FullData)
    X=DataScaler.transform(FullData)

    X_samples = list()
    y_samples = list()
    dates_samples = []

    NumerOfRows = len(X)
    TimeSteps=day2  # next day's Price Prediction is based on last how many past day's prices
    FutureTimeSteps=day1 
    for i in range(TimeSteps , NumerOfRows-FutureTimeSteps , 1):
        x_sample = X[i-TimeSteps:i]
        y_sample = X[i:i+FutureTimeSteps]
        date_sample = dates[i + FutureTimeSteps - 1] 
        X_samples.append(x_sample)
        y_samples.append(y_sample)
        dates_samples.append(date_sample)

    X_data=np.array(X_samples)
    X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
    y_data=np.array(y_samples)
    dates_data = np.array(dates_samples)

    TestingRecords=day1
    X_train=X_data[:-TestingRecords]
    X_test=X_data[-TestingRecords:]
    y_train=y_data[:-TestingRecords]
    y_test=y_data[-TestingRecords:]
    dates_test = dates_data[-TestingRecords:]
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # print('Dates for X_test:', dates_test)

    TimeSteps=X_train.shape[1]
    TotalFeatures=X_train.shape[2]
    # print("Number of TimeSteps:", TimeSteps)
    # print("Number of Features:", TotalFeatures)

    os.chdir(fr"{os.getenv('BASE_DIR')}")
    filename = 'finalized_model'+str(day1)+'.sav'
    regressor = pickle.load(open(filename, 'rb'))

    dates_test = pd.date_range(max_date, periods=day1) #date.today()
    predicted_Price = regressor.predict(X_test)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)
    predicted_df = pd.DataFrame({'Date': dates_test, 'Predicted_Price': predicted_Price[:, 0]})
    # print(predicted_df)

    predicted_Price = regressor.predict(X_test)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)

    #y_test.shape
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

    true_values = DataScaler.inverse_transform(y_test)

    mae = mean_absolute_error(true_values, predicted_Price)
    mse = mean_squared_error(true_values, predicted_Price)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_Price)
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)
    print('R-squared:', r2)
    np.mean(np.abs((true_values - predicted_Price) / true_values)) * 100
    return predicted_df