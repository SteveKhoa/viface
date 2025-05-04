import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))


from protocol.opaque_config import SERVER_BASE_URL, ENDPOINT_LOGIN_1, ENDPOINT_LOGIN_2
from lib import opaque
import base64
import requests


def execute_login_usecase(user_id: str, password: str):
    ids = opaque.Ids(user_id, "server")

    client_pub, client_secret = opaque.CreateCredentialRequest(password)
    client_pub_b64 = base64.b64encode(client_pub).decode("ascii")

    r = requests.post(
        f"{SERVER_BASE_URL}{ENDPOINT_LOGIN_1}",
        data={
            "user_id": user_id,
            "client_pub_b64": client_pub_b64,
        },
    )
    r_json = r.json()
    server_public_b64 = r_json["server_public_b64"]
    server_public = base64.b64decode(server_public_b64)

    try:
        _, client_auth, _ = opaque.RecoverCredentials(
            server_public,
            client_secret,
            "",
            ids,
        )
    except:
        raise Exception("Login failed.")

    client_auth_b64 = base64.b64encode(client_auth).decode("ascii")

    r = requests.post(
        f"{SERVER_BASE_URL}{ENDPOINT_LOGIN_2}",
        data={
            "user_id": user_id,
            "client_auth_b64": client_auth_b64,
        },
    )
    r_json = r.json()
    status = r_json["status"]

    if status != "200":
        detail = r_json["detail"]
        raise Exception(f"Login failed. {detail}")
