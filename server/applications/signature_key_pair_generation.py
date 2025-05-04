from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, BestAvailableEncryption
import sqlite3
from config import SQLITE_PATH, SERVER_SECRET_BASE64

def execute():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    private_key_record = private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=BestAvailableEncryption(SERVER_SECRET_BASE64)
    )



    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("INSERT INTO keypair (public_key, private_key) VALUES (?, ?)", [, private_key_record])