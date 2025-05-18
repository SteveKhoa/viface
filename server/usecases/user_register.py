import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))

from client.adapters import in_user_create


def execute(user_id: str):
    ok = in_user_create.trigger(user_id)

    return ok