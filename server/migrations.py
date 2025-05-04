import sqlite3
from config import SQLITE_PATH


def migrate():
    conn = sqlite3.connect(SQLITE_PATH)

    conn.execute("CREATE TABLE user(id, opaque_rec, UNIQUE(id))")
    conn.execute("CREATE TABLE token(id, tok, UNIQUE(id))")
    conn.execute("CREATE TABLE keypair(public_key, private_key)")

    conn.commit()


if __name__ == "__main__":
    migrate()
