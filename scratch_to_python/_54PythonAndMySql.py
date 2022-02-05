"""
Go to https://dev.mysql.com/downloads/installer/, install mysql-installer-community-<version>.msi
Go to Settings->Python Interpreter->Install a package 'mysql-connector-python'
"""

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="admin",
  password="admin"
)

def getdb():
    return mydb

print(mydb)

mycursor = mydb.cursor()

def getcursor():
    return mycursor

# Create your database
mycursor.execute("CREATE DATABASE mydatabase")

mycursor.execute("SHOW DATABASES")

for x in mycursor:
    print(x)

