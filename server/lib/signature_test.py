import sys
import os
from cryptography.exceptions import InvalidSignature

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))

from server.lib import signature

def test_signature():
    data = b'This is just a dummy data'

    public_key, private_key = signature.generate_public_private_key_pair()
    signed_data = signature.sign(data, private_key)
    signature.validate_signature(signed_data, data, public_key)


def test_signature_invalid_signature():
    data = b'This is just a dummy data'
    invalid_data = b'This is an invalid data'

    try:
        public_key, private_key = signature.generate_public_private_key_pair()
        signed_data = signature.sign(data, private_key)
        signature.validate_signature(signed_data, invalid_data, public_key)
    except InvalidSignature:
        pass
    else:
        raise AssertionError("test_signature_invalid_signature")


if __name__ == "__main__":
    test_signature()
    test_signature_invalid_signature()