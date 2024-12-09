import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('skills.db')
cursor = conn.cursor()

cursor.execute('''INSERT INTO required_skills (skill, source, course, duration, cost) VALUES (?, ?, ?, ?, ?)''', (k, li[0], li[1], li[2], li[3]))


# Commit changes and close connection
conn.commit()
conn.close()