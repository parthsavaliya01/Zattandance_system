import os
import mysql.connector

# Manual parsing of .env file
env_vars = {}
with open("/app/.env", "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("=", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            val = parts[1].strip().strip('"').strip("'")
            env_vars[key] = val

host = env_vars.get("MYSQL_HOST", "172.17.0.1")
port = int(env_vars.get("MYSQL_PORT", 3306))
user = env_vars.get("MYSQL_USER", "root")
password = env_vars.get("MYSQL_PASSWORD", "root@123")
db = env_vars.get("MYSQL_DB", "zydus_attendance")

try:
    conn = mysql.connector.connect(
        host=host, port=port, user=user, password=password, database=db, connection_timeout=3
    )
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM tbl_attendance WHERE doctor_id = 'NAGESHWAR SIR' AND attendance_date = '2026-09-01'")
    rows = cursor.fetchall()
    print("MATCHES:", len(rows))
    for r in rows:
        print(r)

    conn.close()
except Exception as e:
    print("Error:", e)
