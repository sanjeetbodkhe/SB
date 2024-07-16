from flask import Flask, render_template, request, redirect, url_for, session,flash
from fileinput import filename
from werkzeug.utils import secure_filename
from flask import *
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
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
import csv
import threading
from dotenv import load_dotenv

UPLOAD_FOLDER = os.path.join('static', 'uploads')        
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = os.getenv('SECRET_KEY')

# Enter your database connection details below
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD') #Replace with  your database password.
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Intialize MySQL
mysql = MySQL(app)


def configure():
    load_dotenv()

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
        os.chdir(r'C:\Users\Home\Desktop\SBSCAFFORM\static\uploads')
        data = pd.read_csv('aluminium_mini_historical_data.csv',index_col=[0])
        data = data.iloc[::-1]
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
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)
        print('Dates for X_test:', dates_test)

        TimeSteps=X_train.shape[1]
        TotalFeatures=X_train.shape[2]
        print("Number of TimeSteps:", TimeSteps)
        print("Number of Features:", TotalFeatures)

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
        print(predicted_df)

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
        os.chdir(r'C:\Users\Home\Desktop\SBSCAFFORM')
        # save the model to disk
        filename = 'finalized_model'+str(day1)+'.sav'
        pickle.dump(regressor, open(filename, 'wb'))

        np.mean(np.abs((true_values - predicted_Price) / true_values)) * 100
    

# http://localhost:5000/pythonlogin/ - this will be the login page, we need to use both GET and POST requests
@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
# Output message if something goes wrong...
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
                # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            flash("Incorrect username/password!", "danger")
    return render_template('auth/login.html',title="Login")

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    return redirect(url_for('login'))


@app.route('/')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        os.chdir(r'C:\Users\Home\Desktop\SBSCAFFORM')
        if os.path.exists("finalized_model5.sav"):
            np.random.seed(42)
            os.chdir(r'C:\Users\Home\Desktop\SBSCAFFORM\static\uploads')
            data = pd.read_csv('aluminium_mini_historical_data.csv',index_col=[0])
            data = data.iloc[::-1]
            #data.index = pd.to_datetime(data.index)
            dates = data.index.values
            data2 = pd.read_csv('aluminium_mini_historical_data.csv')
            max_date= (dt.datetime.strptime(data2['Date'].max(), "%Y-%m-%d") + dt.timedelta(days=1)).strftime("%Y-%m-%d")

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
            print(X_train.shape)
            print(y_train.shape)
            print(X_test.shape)
            print(y_test.shape)
            print('Dates for X_test:', dates_test)

            TimeSteps=X_train.shape[1]
            TotalFeatures=X_train.shape[2]
            print("Number of TimeSteps:", TimeSteps)
            print("Number of Features:", TotalFeatures)

            os.chdir(r'C:\Users\Home\Desktop\SBSCAFFORM')
            filename = 'finalized_model5.sav'
            regressor = pickle.load(open(filename, 'rb'))

            dates_test = pd.date_range(max_date, periods=5) #date.today()
            predicted_Price = regressor.predict(X_test)
            predicted_Price = DataScaler.inverse_transform(predicted_Price)
            predicted_df = pd.DataFrame({'Date': dates_test, 'Predicted_Price': predicted_Price[:, 0]})
            print(predicted_df)
            print((predicted_df.shape))
            for x in range(5):
                print(predicted_df.iloc[x]['Date'])

            predicted_Price = regressor.predict(X_test)
            predicted_Price = DataScaler.inverse_transform(predicted_Price)

            #y_test.shape
            y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
            # User is loggedin show them the home page
            return render_template('home/home.html', username=session['username'],title="Home",pred=predicted_df,df_len=len(predicted_df),model_status='')
        else:
            return render_template('home/home.html', username=session['username'],title="Home",pred='',df_len=0,model_status='Model is training!')
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))    


@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('auth/profile.html', username=session['username'],title="Profile")
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/home/data')
def data():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home/data.html', username=session['username'],title="Data")
    # User is not loggedin redirect to login page
    return redirect(url_for('data'))

@app.route('/', methods=['POST'])
def getprediction():
    # Check if user is loggedin
    if 'loggedin' in session:
        day1=int(request.form['duration'])
        os.chdir(r'C:\Users\Home\Desktop\SBSCAFFORM')
        if os.path.exists("finalized_model"+str(day1)+".sav"):
            np.random.seed(42)
            #tf.random.set_seed(42)
            os.chdir(r'C:\Users\Home\Desktop\SBSCAFFORM\static\uploads')
            data = pd.read_csv('aluminium_mini_historical_data.csv',index_col=[0])
            print(data.head())
            data = data.iloc[::-1]
            #data.index = pd.to_datetime(data.index)
            dates = data.index.values

            data2 = pd.read_csv('aluminium_mini_historical_data.csv')
            max_date= (dt.datetime.strptime(data2['Date'].max(), "%Y-%m-%d") + dt.timedelta(days=1)).strftime("%Y-%m-%d")
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
            print(X_train.shape)
            print(y_train.shape)
            print(X_test.shape)
            print(y_test.shape)
            print('Dates for X_test:', dates_test)

            TimeSteps=X_train.shape[1]
            TotalFeatures=X_train.shape[2]
            print("Number of TimeSteps:", TimeSteps)
            print("Number of Features:", TotalFeatures)

            os.chdir(r'C:\Users\Home\Desktop\SBSCAFFORM')
            filename = 'finalized_model'+str(day1)+'.sav'
            regressor = pickle.load(open(filename, 'rb'))

            dates_test = pd.date_range(max_date, periods=day1) #date.today()
            predicted_Price = regressor.predict(X_test)
            predicted_Price = DataScaler.inverse_transform(predicted_Price)
            predicted_df = pd.DataFrame({'Date': dates_test, 'Predicted_Price': predicted_Price[:, 0]})
            print(predicted_df)

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
            return render_template('home/home.html', username=session['username'],title="Home",pred=predicted_df,df_len=len(predicted_df),model_status='')
        
        else:
            return render_template('home/home.html', username=session['username'],title="Home",pred='',df_len=0,model_status='Model is training!')
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route("/download_csv")
def download_csv():
    return send_from_directory(UPLOAD_FOLDER,"aluminium_mini_historical_data.csv", as_attachment=True, download_name="aluminium_mini_historical_data.csv")

@app.route('/uploadFile', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        f = request.files.get('file')
        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)
        if(data_filename == ''):
            return render_template('home/home.html', username=session['username'],title="Home",pred='',df_len=0,model_status='')
        else:
            os.chdir(r'C:\Users\Home\Desktop\SBSCAFFORM\static\uploads')
            if os.path.exists("aluminium_mini_historical_data.csv"):
                os.remove("aluminium_mini_historical_data.csv")
        os.chdir(r'C:\Users\Home\Desktop\SBSCAFFORM') 
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],data_filename))       
 
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)        
        print("success in uploading")
        os.chdir(r'C:\Users\Home\Desktop\SBSCAFFORM\static\uploads')
        os.rename(data_filename,"aluminium_mini_historical_data.csv")
        os.chdir(r'C:\Users\Home\Desktop\SBSCAFFORM')
        if os.path.exists("finalized_model5.sav"):
            os.remove("finalized_model5.sav")
        if os.path.exists("finalized_model15.sav"):
            os.remove("finalized_model15.sav")
        if os.path.exists("finalized_model30.sav"):
            os.remove("finalized_model30.sav")

        threading.Thread(target=model_train).start()
        #model_train()      

 
        return render_template('home/home.html', username=session['username'],title="Home",pred='',df_len=0,model_status='Model is training!')
    

if __name__ =='__main__':
    configure()
    app.run(debug=True)