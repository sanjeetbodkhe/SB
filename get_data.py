from flask_mysqldb import MySQL
import app
import MySQLdb.cursors
import pandas as pd
import os



def fetch_alu_mcx():
    cursor = app.mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(f"SELECT * FROM {os.getenv('MYSQL_DB')}.data")
    data = pd.DataFrame(cursor.fetchall())
    data.set_index('Date', inplace=True)
    return data

def fetch_alu_mcx2():
    cursor = app.mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(f"SELECT * FROM {os.getenv('MYSQL_DB')}.data1")
    data = pd.DataFrame(cursor.fetchall())
    data.set_index('Date', inplace=True)
    data=data.sort_values(by=['Date'])
    return data