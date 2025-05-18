import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))

from client.usecases import enroll

def trigger(user_id: str):
    ok = enroll.execute(user_id)

    return ok