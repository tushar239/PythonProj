"""
Go to https://dev.mysql.com/downloads/installer/, install mysql-installer-community-<version>.msi
Go to Settings->Python Interpreter->Install a package 'mysql-connector-python'
"""

import mysql.connector

connection = mysql.connector.connect(
  host="localhost",
  user="admin",
  password="admin"
)

print(connection)

cursor = connection.cursor()

# Create your database
cursor.execute("CREATE DATABASE mydatabase")

cursor.execute("SHOW DATABASES")

for x in cursor:
    print(x)

