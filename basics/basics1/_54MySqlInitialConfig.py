import mysql.connector

"""
The connection has one purpose: controlling access to the database. 
The cursor has one purpose: keeping track of where we are in the database so that if several programs are accessing the database at the same time, the database can keep track of who is trying to do what.

A cursor is a temporary work area created in the system memory when a SQL statement is executed. A cursor contains information on a select statement and the rows of data accessed by it. 
This temporary work area is used to store the data retrieved from the database, and manipulate this data.
This cursor is on DB side and records are fetched one by one lazily, unless you say connection.cursor(buffered=true). When you buffer it, all the records will be fetched from DB cursor and kept in memory.
"""

connection = mysql.connector.connect(
    host="localhost",
    user="admin",
    password="admin",
    database="mydatabase"
)

cursor = connection.cursor()
