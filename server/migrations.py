import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import sqlite3
from server.constant import SQLITE_PATH


def migrate():
    conn = sqlite3.connect(SQLITE_PATH)

    try:
        conn.execute("CREATE TABLE user(id, user_id, opaque_rec, UNIQUE(id))")
        conn.execute("CREATE TABLE token(id, tok, UNIQUE(id))")
        conn.execute("CREATE TABLE access_token_request(id, domain, user_id, UNIQUE(id))")
    except sqlite3.OperationalError:
        print("migrate: warning, duplicated tables encountered.")
        pass  # we don't handle duplicate tables

    conn.commit()


if __name__ == "__main__":
    migrate()
