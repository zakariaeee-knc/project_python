import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="2929",      # change this
        database="instagrame_data"   # change this
    )
