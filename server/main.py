import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import sqlite3
import json
import base64
from os import urandom
from typing_extensions import Annotated
from fastapi import FastAPI, Form, Depends
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from server.constant import SQLITE_PATH, SERVER_SECRET_BASE64, SIGNATURE_PUBLIC_KEY
from server.usecases import access_token_get_consent, access_token_push, user_register
from lib import opaque
import time
import jwt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


@app.post("/opaque/registration/1")
def registration_step_1(
    user_id: Annotated[str, Form()],
    client_pub_b64: Annotated[str, Form()],
):

    client_pub = base64.b64decode(client_pub_b64)

    server_session_secret, server_pub = opaque.CreateRegistrationResponse(
        client_pub,
        base64.b64decode(SERVER_SECRET_BASE64),
    )

    server_pub_b64 = base64.b64encode(server_pub).decode("ascii")

    with open(f"session_{user_id}.json", "w") as f:
        server_session_secret_b64 = base64.b64encode(server_session_secret).decode(
            "ascii"
        )
        data = {"server_session_secret_b64": server_session_secret_b64}
        json.dump(data, f)

    return {"status": "200", "server_pub_b64": server_pub_b64}


@app.post("/opaque/registration/2")
def registration_step_2(
    user_id: Annotated[str, Form()],
    rec0_b64: Annotated[str, Form()],
):

    rec0 = base64.b64decode(rec0_b64)

    session_file = f"session_{user_id}.json"
    with open(session_file, "r") as f:
        data = json.load(f)
        server_session_secret_b64 = data["server_session_secret_b64"]
        server_session_secret = base64.b64decode(server_session_secret_b64)
    os.remove(session_file)

    rec1 = opaque.StoreUserRecord(server_session_secret, rec0)  # server 2

    conn = sqlite3.connect(SQLITE_PATH)
    try:
        conn.execute("INSERT INTO user (id, opaque_rec) VALUES (?, ?)", [user_id, rec1])
        resp = {"status": "200"}
    except sqlite3.IntegrityError:
        resp = {"status": "400", "detail": "Account already created."}
    conn.commit()

    return resp


@app.post("/opaque/login/1")
def login(
    user_id: Annotated[str, Form()],
    client_pub_b64: Annotated[str, Form()],
):

    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.execute("SELECT * FROM user WHERE id=?", [user_id])
    user = cur.fetchone()

    if user is None:
        return {"status": "401", "detail": "Account not registrated."}

    user_id = user[0]
    user_rec = user[1]

    client_pub = base64.b64decode(client_pub_b64)

    ids = opaque.Ids(user_id, "server")

    server_public, _, server_session_secret = opaque.CreateCredentialResponse(
        client_pub, user_rec, ids, ""
    )
    server_public_b64 = base64.b64encode(server_public).decode("ascii")

    with open(f"session_{user_id}.json", "w") as f:
        server_session_secret_b64 = base64.b64encode(server_session_secret).decode(
            "ascii"
        )
        data = {"server_session_secret_b64": server_session_secret_b64}
        json.dump(data, f)

    return {"status": "200", "server_public_b64": server_public_b64}


@app.post("/opaque/login/2")
def login(
    user_id: Annotated[str, Form()],
    client_auth_b64: Annotated[str, Form()],
):

    client_auth = base64.b64decode(client_auth_b64)

    session_file = f"session_{user_id}.json"
    with open(session_file, "r") as f:
        data = json.load(f)
        server_session_secret_b64 = data["server_session_secret_b64"]
        server_session_secret = base64.b64decode(server_session_secret_b64)
    os.remove(session_file)

    try:
        opaque.UserAuth(server_session_secret, client_auth)
    except:
        resp = {"status": "401", "detail": "Login failed"}
        return resp

    # Create authentication token for authenticator (client)
    token_id = base64.b32encode(urandom(4)).decode("ascii")
    token = base64.b32encode(urandom(32)).decode("ascii")

    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("INSERT INTO token (id, tok) VALUES (?, ?)", [token_id, token])

    token_b64 = base64.b64encode(token).decode("ascii")

    resp = {"status": "200", "token_b64": token_b64}
    return resp


@app.get("/token")
async def request_token(domain: str, user_id: str):
    """
    Asynchronously create an access token request. The access token request will be pushed back to client via server send event.
    """

    log_time_start = time.time()

    json_resp = access_token_get_consent.execute(domain, user_id)

    status = json_resp["status"]
    msg = json_resp["msg"]

    if status == "200":

        test_access_token = {
            "iss": "viface.com",
            "sub": "viface://testuser",
            "exp": 9999999999,
            "scopes": ["catalog:read", "cart:read,write"],
        }

        access_token = jwt.encode(test_access_token, SERVER_SECRET_BASE64, algorithm="HS256")
    else:
        access_token = ""

    log_time_end = time.time()

    print("request_token:", log_time_end - log_time_start, "s")

    return {"status": status, "msg": msg, "access_token": access_token}


@app.get("/register")
async def register(user_id: str):
    log_time_start = time.time()

    ok = user_register.execute(user_id)

    if ok:
        status = "200"
        msg = "User created successfully"
    else:
        status = "500"
        msg = "Registration failed. Unexpected error on the server"

    log_time_end = time.time()

    print("register:", log_time_end - log_time_start, "s")

    return {"status": status, "msg": msg}


@app.get("/resource")
async def resource(access_token: str):
    log_time_start = time.time()

    data = "You are my TREASURE!"

    try:
        decoded_jwt = jwt.decode(access_token, SERVER_SECRET_BASE64, algorithms=["HS256"])

        status = "200"
        msg = "Success"
    except:
        status = "401"
        msg = "Unauthorized"

    log_time_end = time.time()

    print("resource:", log_time_end - log_time_start, "s")

    return {"status": status, "msg": msg, "data": data}


@app.get("/")
def read_item():
    return {"status": "200", "detail": "hello, world!"}
