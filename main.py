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
from model import def_model, model_train, get_prediction

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
        cursor.execute(f"SELECT * FROM {os.getenv('MYSQL_DB')}.accounts WHERE username = '{username}' AND password = '{password}'")
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
        os.chdir(fr"{os.getenv('BASE_DIR')}")
        if os.path.exists("finalized_model5.sav"):
            predicted_df=def_model()
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
        os.chdir(fr"{os.getenv('BASE_DIR')}")
        if os.path.exists("finalized_model"+str(day1)+".sav"):
            predicted_df=get_prediction(day1)
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
            os.chdir(fr"{os.getenv('BASE_DIR')}\static\uploads")
            if os.path.exists("aluminium_mini_historical_data.csv"):
                os.remove("aluminium_mini_historical_data.csv")
        os.chdir(fr"{os.getenv('BASE_DIR')}") 
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],data_filename))       
 
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)        
        print("success in uploading")
        os.chdir(fr"{os.getenv('BASE_DIR')}\static\uploads")
        os.rename(data_filename,"aluminium_mini_historical_data.csv")
        os.chdir(fr"{os.getenv('BASE_DIR')}")
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