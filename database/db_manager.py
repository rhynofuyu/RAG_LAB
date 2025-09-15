import sqlite3
import hashlib
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage


class DatabaseManager:
    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (username TEXT PRIMARY KEY, password TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT,
                      message_content TEXT,
                      message_type TEXT,
                      timestamp DATETIME,
                      FOREIGN KEY (username) REFERENCES users (username))''')
        conn.commit()
        conn.close()

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username, password):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            hashed_password = self.hash_password(password)
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def verify_user(self, username, password):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        hashed_password = self.hash_password(password)
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
        result = c.fetchone()
        conn.close()
        return result is not None

    def change_password(self, username, old_password, new_password):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            if self.verify_user(username, old_password):
                c.execute("UPDATE users SET password = ? WHERE username = ?", 
                         (self.hash_password(new_password), username))
                conn.commit()
                return True
            else:
                return False
        finally:
            conn.close()

    def save_message_to_db(self, username, message_content, message_type):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now()
        c.execute("INSERT INTO chat_history (username, message_content, message_type, timestamp) VALUES (?, ?, ?, ?)",
                  (username, message_content, message_type, timestamp))
        conn.commit()
        conn.close()

    def load_chat_history_with_window(self, username, window_size):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        limit = window_size * 2
        c.execute("""
            SELECT message_content, message_type 
            FROM chat_history 
            WHERE username = ? 
            ORDER BY timestamp DESC
            LIMIT ?
        """, (username, limit))
        results = c.fetchall()
        conn.close()
        
        results.reverse()

        chat_messages = []
        for content, msg_type in results:
            if msg_type == 'user':
                chat_messages.append(HumanMessage(content=content))
            else:
                chat_messages.append(AIMessage(content=content))
        return chat_messages

    def clear_user_chat_history(self, username):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM chat_history WHERE username = ?", (username,))
        conn.commit()
        conn.close()
