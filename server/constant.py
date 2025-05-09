import os
import base64
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat

SQLITE_PATH = os.getenv("SERVER_SQLITE_PATH")
SERVER_SECRET_BASE64 = os.getenv("SERVER_SECRET_BASE64")
SERVER_SECRET = base64.b64decode(SERVER_SECRET_BASE64)
PRIVATE_KEY_ENCODING = os.getenv("SERVER_PRIVATE_KEY_ENCODING")
PRIVATE_KEY_FORMAT = os.getenv("SERVER_PRIVATE_KEY_FORMAT")
PUBLIC_KEY_FORMAT = os.getenv("SERVER_PUBLIC_KEY_FORMAT")

# To solve problem "TypeError: encoding must be an item from the Encoding enum"
ENUM_PRIVATE_KEY_ENCODING = Encoding(PRIVATE_KEY_ENCODING)
ENUM_PRIVATE_KEY_FORMAT = PrivateFormat(PRIVATE_KEY_FORMAT)
ENUM_PUBLIC_KEY_FORMAT = PublicFormat(PUBLIC_KEY_FORMAT)

SIGNATURE_PRIVATE_KEY = base64.b64decode(os.getenv("SERVER_SIGNATURE_PUBLIC_KEY_BASE64"))
SIGNATURE_PUBLIC_KEY = base64.b64decode(os.getenv("SERVER_SIGNATURE_PRIVATE_KEY_BASE64"))