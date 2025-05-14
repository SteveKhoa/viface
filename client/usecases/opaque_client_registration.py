import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from client.constant import SERVER_BASE_URL, ENDPOINT_REGISTRATION_1, ENDPOINT_REGISTRATION_2
from lib import opaque
import base64
import requests


def execute_registration_usecase(user_id: str, password: str) -> dict:
    ids = opaque.Ids(user_id, "server")

    # Public messages exchange
    client_secret, client_pub = opaque.CreateRegistrationRequest(password)
    client_pub_b64 = base64.b64encode(client_pub).decode("ascii")

    r = requests.post(
        f"{SERVER_BASE_URL}{ENDPOINT_REGISTRATION_1}",
        data={
            "user_id": user_id,
            "client_pub_b64": client_pub_b64,
        },
    )
    r_json = r.json()
    server_pub_b64 = r_json["server_pub_b64"]
    server_pub = base64.b64decode(server_pub_b64)

    rec0, _ = opaque.FinalizeRequest(
        client_secret,
        server_pub,
        ids,
    )
    rec0_b64 = base64.b64encode(rec0).decode("ascii")

    r = requests.post(
        f"{SERVER_BASE_URL}{ENDPOINT_REGISTRATION_2}",
        data={
            "user_id": user_id,
            "rec0_b64": rec0_b64,
        },
    )
    r_json = r.json()
    status = r_json["status"]

    if status != "200":
        detail = r_json["detail"]
        raise Exception(f"Registration failed. {detail}")
    
    return r_json
