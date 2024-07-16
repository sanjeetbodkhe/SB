##Read Me

#Open app folder after cloning and start venv:

python3 -m venv venv

#Activate environment:

venv\Scripts\activate

#Install dependencies:

pip install -r requirements.txt

#Database:

Make changes according to .sql file

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