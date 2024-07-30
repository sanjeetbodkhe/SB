import requests 
from bs4 import BeautifulSoup 
import csv 
import datetime
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
from urllib.parse import quote_plus


# Connect to the database
engine = create_engine(f'mysql+mysqlconnector://{os.getenv("MYSQL_USER")}:%s@{os.getenv("MYSQL_HOST")}/{os.getenv("MYSQL_DB")}' % quote_plus(os.getenv("MYSQL_PASSWORD")))

# Test the connection
connection = engine.connect()


#url = r"https://in.investing.com/commodities/aluminium-mini-historical-data?end_date=1716982199&st_date=1611858600"

# Headers to mimic a browser visit
#headers = {
#    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
#}

def alu_mcx_db():
    #Data from https://in.investing.com/commodities/aluminium-mini-historical-data
    url = open("C:/Users/Home/Desktop/SBSCAFFORM/sites/InvestingcomIndia.html", encoding="utf8")
    soup = BeautifulSoup(url, 'html.parser')
    table = soup.find('table', {'class': 'freeze-column-w-1 w-full overflow-x-auto text-xs leading-4'})
    headers = [header.text.strip() for header in table.find_all('th')]
    for i in range(len(headers)):
        headers[i]=''.join(e for e in headers[i] if e.isalnum())
    rows = table.find_all('tr')
    # Extract data from each row
    data = []
    for row in rows[1:]:  # Skip the header row
        cols = row.find_all('td')
        cols = [col.text.strip() for col in cols]
        data.append(cols)
    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data, columns=headers)
    df['Date'] = pd.to_datetime(df.Date,format='%d-%m-%Y')
    df.rename(columns = {'Change':'Chng'}, inplace = True)
    df.to_sql(con=connection, name='data1',index=df['Date'],index_label='Date', if_exists='replace')
    #print(df)
    return df
#alu_mcx_db()
# sql = "select * from data1"
# df2 = pd.read_sql(sql,con=engine)
# print(df2)

def lme_alu():
    #Data from https://www.westmetall.com/en/markdaten.php?action=table&field=LME_Al_cash#y2024
    url = open("C:/Users/Home/Desktop/SBSCAFFORM/sites/LME_ALU.html", encoding="utf8")
    soup = BeautifulSoup(url, 'html.parser')
    table = soup.find_all('table')
    headers = [header.text.strip() for header in table[0].find_all('th')]
    headers=headers[:4]
    all_df=pd.DataFrame(columns=headers)    
    for i in range(0,4):
        rows = table[i].find_all('tr')
        # Extract data from each row
        data = []
        for row in rows[1:]:  # Skip the header row
            cols = row.find_all('td')
            cols = [col.text.strip() for col in cols]
            data.append(cols)
        # Create a DataFrame from the extracted data
        df = pd.DataFrame(data, columns=headers)
        frames = [all_df, df]
        all_df=pd.concat(frames)    
    all_df['date'] = pd.to_datetime(all_df.date,format='%d. %B %Y')
    all_df.rename(columns = {'date':'Date','LME Aluminium 3-month':'LME_ALU_PRICE'}, inplace = True)
    all_df=all_df.drop(columns=['LME Aluminium Cash-Settlement', 'LME Aluminium stock'], axis=1)
    #print(all_df)
    return all_df
#lme_alu()


def usd_inr():
    #Data from https://in.investing.com/currencies/usd-inr-historical-data
    url = open("C:/Users/Home/Desktop/SBSCAFFORM/sites/USDINRInvestingcomIndia.html", encoding="utf8")
    soup = BeautifulSoup(url, 'html.parser')
    table = soup.find('table', {'class': 'freeze-column-w-1 w-full overflow-x-auto text-xs leading-4'})
    headers = [header.text.strip() for header in table.find_all('th')]
    rows = table.find_all('tr')
    # Extract data from each row
    data = []
    for row in rows[1:]:  # Skip the header row
        cols = row.find_all('td')
        cols = [col.text.strip() for col in cols]
        data.append(cols)
    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data, columns=headers)
    df['Date'] = pd.to_datetime(df.Date,format='%d-%m-%Y')
    df=df.drop(columns=['Open', 'High', 'Low', 'Vol.', 'Change %'],axis=1)    
    df.rename(columns = {'Price':'usd_inr_conv'}, inplace = True)
    #print(df)
    return df
#usd_inr()

def inflation():
    #Data from https://www.moneycontrol.com/economic-calendar/india-inflation-rate-yoy/5128767
    url = open("C:/Users/Home/Desktop/SBSCAFFORM/sites/IndiaInflationMoneycontrol.html", encoding="utf8")
    soup = BeautifulSoup(url, 'html.parser')
    table = soup.find('table', {'id': 'hist_tbl'})
    headers = [header.text.strip() for header in table.find_all('th')]
    rows = table.find_all('tr')
    # Extract data from each row
    data = []
    for row in rows[1:]:  # Skip the header row
        cols = row.find_all('td')
        cols = [col.text.strip() for col in cols]
        data.append(cols)
    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data, columns=headers)
    df['Date'] = pd.to_datetime(df.Date,format='%b %d, %Y')
    df = df[df.Actual != '-']
    df=df.drop(columns=['Time', 'Reference', 'Previous', 'Consensus'])
    df['year'] = pd.DatetimeIndex(df['Date']).year
    df['month'] = pd.DatetimeIndex(df['Date']).month    
    df.rename(columns = {'Actual':'Inflation'}, inplace = True)
    #print(df)
    return df
#inflation()

def crude_oil():
    # Data from https://www.nasdaq.com/market-activity/commodities/cl-nmx/historical?page=1&rows_per_page=10&timeline=y5
    os.chdir(fr"{os.getenv('BASE_DIR')}\sites")
    df = pd.read_csv('Crude oil.csv')
    df['Date'] = pd.to_datetime(df.Date,format='%m/%d/%Y')
    df=df.drop(columns=['Volume', 'Open', 'High', 'Low'])
    df.rename(columns = {'Price':'Crude_Price'}, inplace = True)
    #print(df)
    return df
#crude_oil()

#collaborate all data and predict using formula
def nalco_pred():
    df_lme_alu = lme_alu()
    df_usd_inr=usd_inr()
    df_inflat=inflation()
    df_crude=crude_oil()
    df=pd.merge(df_lme_alu, df_usd_inr, on='Date', how='inner')
    df['year'] = pd.DatetimeIndex(df['Date']).year
    df['month'] = pd.DatetimeIndex(df['Date']).month
    df=pd.merge(df, df_inflat, on=['year','month'], how='inner')
    df=df.drop(columns=['year', 'month','Date_y'])
    df.rename(columns = {'Date_x':'Date'}, inplace = True)
    df=pd.merge(df, df_crude, on='Date', how='inner')
    df['LME_ALU_PRICE'] = df['LME_ALU_PRICE'].map(lambda LME_ALU_PRICE: float(LME_ALU_PRICE.replace(",", "")))
    df['usd_inr_conv'] = df['usd_inr_conv'].map(lambda usd_inr_conv: float(usd_inr_conv))
    df['Inflation'] = df['Inflation'].map(lambda Inflation: float(Inflation.replace("%", "")))
    df['Crude_Price'] = df['Crude_Price'].map(lambda Crude_Price: float(Crude_Price))
    df = df.assign(Nalco_Pred = round(((df.LME_ALU_PRICE * df.usd_inr_conv * 45) + (df.Inflation * 0.8) + (df.Crude_Price*0.4) + 12500),2))
    #print(df)
    return df

print(nalco_pred())
#alu_mcx_db()