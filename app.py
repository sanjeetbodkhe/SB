from flask import Flask, render_template, request, redirect, url_for, session,flash
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
import threading
from dotenv import load_dotenv
import model
import get_data


app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = os.getenv('SECRET_KEY')

# Enter your database connection details below
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD') #Replace with  your database password.
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')

all_models_trained=True


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

@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('auth/profile.html', username=session['username'],title="Profile")
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


@app.route('/')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        os.chdir(fr"{os.getenv('BASE_DIR')}")
        if(os.path.exists("finalized_model30.sav")):
            all_models_trained=True
        else:
            all_models_trained=False
        if os.path.exists("finalized_model5.sav"):
            predicted_df=model.get_prediction(5)
            # User is loggedin show them the home page
            return render_template('home/home.html', username=session['username'],title="Home",pred=predicted_df,df_len=len(predicted_df),model_status='',all_models_flag=all_models_trained)
        else:
            return render_template('home/home.html', username=session['username'],title="Home",pred='',df_len=0,model_status='Model is training!',all_models_flag=all_models_trained)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))    

@app.route('/', methods=['POST'])
def getprediction():
    # Check if user is loggedin
    if 'loggedin' in session:
        day1=int(request.form['duration'])
        os.chdir(fr"{os.getenv('BASE_DIR')}")
        if(os.path.exists("finalized_model30.sav")):
            all_models_trained=True
        else:
            all_models_trained=False
        if os.path.exists("finalized_model"+str(day1)+".sav"):
            predicted_df=model.get_prediction(day1)
            return render_template('home/home.html', username=session['username'],title="Home",pred=predicted_df,df_len=len(predicted_df),model_status='',all_models_flag=all_models_trained)        
        else:
            return render_template('home/home.html', username=session['username'],title="Home",pred='',df_len=0,model_status='Model is training!',all_models_flag=all_models_trained)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route("/train_model", methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        os.chdir(fr"{os.getenv('BASE_DIR')}")
        if os.path.exists("finalized_model5.sav"):
            os.remove("finalized_model5.sav")
        if os.path.exists("finalized_model15.sav"):
            os.remove("finalized_model15.sav")
        if os.path.exists("finalized_model30.sav"):
            os.remove("finalized_model30.sav")

        os.chdir(fr"{os.getenv('BASE_DIR')}")
        if(os.path.exists("finalized_model30.sav")):
            all_models_trained=True
        else:
            all_models_trained=False
        
        data = get_data.fetch_alu_mcx()
        threading.Thread(target=model.model_train,args=(data,)).start()
        #model_train()
        return render_template('home/home.html', username=session['username'],title="Home",pred='',df_len=0,model_status='Model is training!',all_models_flag=all_models_trained)
    
@app.route('/home/data')
def data():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        df_table=get_data.fetch_alu_mcx()
        df_table_len= len(df_table)
        df_table=df_table.reset_index(drop=False)
        df_table=df_table.sort_values(by=['Date'], ascending=False)
        return render_template('home/data.html', username=session['username'],title="Data",users=df_table,df_table_len_f=df_table_len)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/add', methods =["GET", "POST"])
def add():
    if request.method == "POST":
       date = request.form.get("date")
       price = request.form.get("price")
       open = request.form.get("open")
       high = request.form.get("high")
       low = request.form.get("low")
       vol = request.form.get("vol")
       change = request.form.get("change")
       cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
       cursor.execute(f"SELECT * FROM {os.getenv('MYSQL_DB')}.alu_mcx_data WHERE date = '{date}'")
       data = cursor.fetchone()
       if data == None:
           cursor.execute(f"INSERT INTO {os.getenv('MYSQL_DB')}.alu_mcx_data VALUES('{date}',{price},{open},{high},{low},'{vol}','{change}')")
       else:
           cursor.execute((f"UPDATE {os.getenv('MYSQL_DB')}.alu_mcx_data SET Price = {price} WHERE date='{date}' "))
           cursor.execute((f"UPDATE {os.getenv('MYSQL_DB')}.alu_mcx_data SET Open={open} WHERE date='{date}' "))
           cursor.execute((f"UPDATE {os.getenv('MYSQL_DB')}.alu_mcx_data SET High={high} WHERE date='{date}' "))
           cursor.execute((f"UPDATE {os.getenv('MYSQL_DB')}.alu_mcx_data SET Vol='{vol}' WHERE date='{date}' "))
           cursor.execute((f"UPDATE {os.getenv('MYSQL_DB')}.alu_mcx_data SET chnge='{change}' WHERE date='{date}' "))
       mysql.connection.commit()
    df_table=get_data.fetch_alu_mcx()
    df_table_len= len(df_table)
    df_table=df_table.reset_index(drop=False)
    df_table=df_table.sort_values(by=['Date'], ascending=False)
    return render_template('home/data.html', username=session['username'],title="Data",users=df_table,df_table_len_f=df_table_len)
    
@app.route('/delete', methods =["GET", "POST"])
def delete():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       date = request.form.get("date")
       # getting input with name = lname in HTML form
       cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
       cursor.execute(f"DELETE FROM {os.getenv('MYSQL_DB')}.alu_mcx_data WHERE date = '{date}'") 
       mysql.connection.commit()
    df_table=get_data.fetch_alu_mcx()
    df_table_len= len(df_table)
    df_table=df_table.reset_index(drop=False)
    df_table=df_table.sort_values(by=['Date'], ascending=False)
    return render_template('home/data.html', username=session['username'],title="Data",users=df_table,df_table_len_f=df_table_len)

if __name__ =='__main__':
    configure()
    app.run(debug=True)