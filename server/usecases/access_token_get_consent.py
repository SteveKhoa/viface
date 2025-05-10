import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))

import sqlite3
from server.constant import SQLITE_PATH
from client.adapters import in_consent_get


def execute(domain: str, user_id: str):
    # conn = sqlite3.connect(SQLITE_PATH)

    # conn.execute("INSERT INTO access_token_request (domain, user_id) VALUES (?, ?)", domain, user_id)

    # Little hack for fast demonstration
    # Call the function in `client` directly
    json_resp = in_consent_get.trigger(user_id)

    if json_resp["status"] == "200":
        return True
    else:
        return False