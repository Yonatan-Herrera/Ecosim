import sqlite3
from flask import Flask, jsonify

app = Flask(__name__)

# Define the database path
DATABASE = 'ecosim.db'

def get_db_conn():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    # Return rows as dictionaries instead of tuples
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/api/latest_stats")
def latest_stats():
    """Provides the most recent KPI row from the database."""
    conn = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        
        # Query for the row with the highest 'tick' value
        cursor.execute("SELECT * FROM kpis ORDER BY tick DESC LIMIT 1")
        
        latest_row = cursor.fetchone()
        
        if latest_row:
            # Convert the sqlite3.Row object to a standard dictionary
            # and then return it as JSON
            return jsonify(dict(latest_row))
        else:
            # Handle case where the table is empty
            return jsonify({"error": "No data found in kpis table"}), 404
            
    except sqlite3.Error as e:
        # Handle potential database errors
        print(f"Database error: {e}")
        return jsonify({"error": "Database error occurred"}), 500
    except Exception as e:
        # Handle other potential errors
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("Starting Flask server at http://127.0.0.1:5000/api/latest_stats")
    app.run(debug=True, port=5000)