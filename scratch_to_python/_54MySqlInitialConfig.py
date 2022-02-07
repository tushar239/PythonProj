import mysql.connector

"""
The connection has one purpose: controlling access to the database. 
The cursor has one purpose: keeping track of where we are in the database so that if several programs are accessing the database at the same time, the database can keep track of who is trying to do what.

A cursor is a temporary work area created in the system memory when a SQL statement is executed. A cursor contains information on a select statement and the rows of data accessed by it. 
This temporary work area is used to store the data retrieved from the database, and manipulate this data.
"""

connection = mysql.connector.connect(
    host="localhost",
    user="admin",
    password="admin",
    database="mydatabase"
)

cursor = connection.cursor()
