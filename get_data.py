from flask_mysqldb import MySQL
import main
import MySQLdb.cursors
import pandas as pd
import os



def fetch_alu_mcx():
    cursor = main.mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(f"SELECT * FROM {os.getenv('MYSQL_DB')}.data")
    data = pd.DataFrame(cursor.fetchall())
    data.set_index('Date', inplace=True)
    return data