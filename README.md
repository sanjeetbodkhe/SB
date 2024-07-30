##Read Me

#Open app folder after cloning and start venv:

python3 -m venv venv

#Activate environment:

venv\Scripts\activate

#Install dependencies:

pip install -r requirements.txt

#Database:

Make changes according to .sql file

#Create .env file with parameters like these:
SECRET_KEY = 1a2b3c4d5e6d7g8h9i10
MYSQL_HOST =localhost
MYSQL_USER = root
MYSQL_PASSWORD=Akalpit@1
MYSQL_DB = loginapp
BASE_DIR=C:\Users\Home\Desktop\SBSCAFFORM

#Nalco Prediction formula:
Nalco IA10 Aluminium Ingot Price (Rs/ton) ≈ (LME Aluminium Rate (USD/ton) x USD-INR Exchange Rate x 45) + (INR Inflation Rate x 0.8) + (Crude Oil Price (USD/barrel) x 0.4) + 12,500

#To run the app:

venv\Scripts\activate
set FLASK_APP=main.py
set FLASK_DEBUG=1
set FLASK_ENV=development
flask run

#To restart nginx server:

sudo service nginx restart
sudo service app restart
sudo systemctl restart nginx

#To check server logs:

sudo tail -F /var/log/nginx/error.log 



##Reference links:

Login Page:https://github.com/HarunMbaabu/Login-System-with-Python-Flask-and-MySQL/blob/master/README.md

Server Deployment:https://www.codewithharry.com/blogpost/flask-app-deploy-using-gunicorn-nginx/