import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))

from client.usecases import consent

def trigger(user_id: str):
    is_consent, msg = consent.execute(user_id)

    if is_consent:
        return {"status": "200", "msg": msg}
    else:
        return {"status": "401", "msg": msg}