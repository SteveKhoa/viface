from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    load_pem_private_key,
    load_pem_public_key,
)
from server.constant import (
    SERVER_SECRET,
    ENUM_PRIVATE_KEY_ENCODING,
    ENUM_PRIVATE_KEY_FORMAT,
    ENUM_PUBLIC_KEY_FORMAT,
)


def generate_public_private_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )

    private_key_record = private_key.private_bytes(
        encoding=ENUM_PRIVATE_KEY_ENCODING,
        format=ENUM_PRIVATE_KEY_FORMAT,
        encryption_algorithm=BestAvailableEncryption(SERVER_SECRET),
    )

    public_key = private_key.public_key()

    public_key_record = public_key.public_bytes(
        encoding=ENUM_PRIVATE_KEY_ENCODING,
        format=ENUM_PUBLIC_KEY_FORMAT,
    )

    return (public_key_record, private_key_record)


def sign(data: bytes, private_key_record: bytes):
    private_key = load_pem_private_key(
        data=private_key_record,
        password=SERVER_SECRET,
        backend=default_backend(),
    )

    return private_key.sign(
        data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256()
    )


def validate_signature(signature: bytes, payload: bytes, public_key_record: bytes):
    public_key = load_pem_public_key(
        data=public_key_record,
        backend=default_backend(),
    )

    public_key.verify(
        signature,
        payload,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256()
    )
