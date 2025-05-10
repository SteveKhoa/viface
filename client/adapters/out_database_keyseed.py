import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))

from client.constant import ENROLL_USER_VAULT_DIR
import base64
import json


def save(keyseed, user_id: str):
    (cipher, mask, nonce) = keyseed

    with open(os.path.join(ENROLL_USER_VAULT_DIR, f"{user_id}.json"), "w") as f:
        data = {
            "cipher": serialize(cipher),
            "mask": serialize(mask),
            "nonce": serialize(nonce),
        }
        json.dump(data, f)


def get(user_id: str):
    try:
        with open(os.path.join(ENROLL_USER_VAULT_DIR, f"{user_id}.json"), "r") as f:
            data = json.load(f)

            cipher_str = data["cipher"]
            mask_str = data["mask"]
            nonce_str = data["nonce"]

            cipher = deserialize(cipher_str)
            mask = deserialize(mask_str)
            nonce = deserialize(nonce_str)

        ok = True
    except FileNotFoundError:

        ok = False

    return ((cipher, mask, nonce), ok)


def serialize(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def deserialize(string: str) -> bytes:
    return base64.b64decode(string)
