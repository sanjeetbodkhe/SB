import requests 
from bs4 import BeautifulSoup 
import csv 
import datetime
import pandas as pd

#url = r"https://in.investing.com/commodities/aluminium-mini-historical-data?end_date=1716982199&st_date=1611858600"

# Headers to mimic a browser visit
#headers = {
#    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
#}

url = open("C:/Users/Home/Desktop/SBSCAFFORM/InvestingcomIndia.html", encoding="utf8")

soup = BeautifulSoup(url, 'html.parser')

table = soup.find('table', {'class': 'freeze-column-w-1 w-full overflow-x-auto text-xs leading-4'})
print(soup)

headers = [header.text.strip() for header in table.find_all('th')]

rows = table.find_all('tr')

# Extract data from each row
data = []
for row in rows[1:]:  # Skip the header row
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    data.append(cols)

print(data)

# Create a DataFrame from the extracted data
df = pd.DataFrame(data, columns=headers)

#df['Date'] = pd.to_datetime(df.Date,format='%b %d, %Y')

df['Date'] = pd.to_datetime(df.Date,format='%d-%m-%Y')

#df['Date1'] = df['Date'].dt.strftime('%d/%m/%Y')
# Save the DataFrame to a CSV file
df.to_csv('aluminium_mini_historical_data1.csv', index=False)

print(df)

