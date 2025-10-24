#For every simulation tick, the backend calls log_simulation_turn() to record data. 

import sqlite3

#Function init_db: sets up the database and table
def init_db():
   
    conn = sqlite3.connect("ecosim.db")  #connect to "ecosim.db"
    c = conn.cursor() # create cursor object let us run SQL commands
    c.execute("""
        CREATE TABLE IF NOT EXISTS kpis (
            tick INTEGER PRIMARY KEY,
            gdp REAL,
            unemployment_rate REAL,
            total_spending REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_simulation_turn(tick, gdp, unemployment_rate, total_spending):
    conn = sqlite3.connect("ecosim.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO kpis (tick, gdp, unemployment_rate, total_spending)
        VALUES (?, ?, ?, ?)
    """, (tick, gdp, unemployment_rate, total_spending))
    conn.commit()
    conn.close()