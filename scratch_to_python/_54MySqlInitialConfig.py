import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="admin",
    password="admin",
    database="mydatabase"
)

mycursor = mydb.cursor()
