from lib.biocryp import keygen
import os

keygen_fuzzy_extractor = keygen.MaskThenLockFuzzyExtractor(
    input_length=512,
    key_length=16,
    mask_length=284,
    nonce_len=1,
)

DIRECTORY_CURRENT = os.getenv("DIRECTORY_CURRENT")
USER_ID = os.getenv("CLIENT_USER_ID")
IMAGE_PATH = os.getenv("CLIENT_IMAGE_PATH")
BIOCRYP_HELPER_DIR = os.getenv("CLIENT_DIRECTORY_BIOCRYP_HELPER")

SERVER_BASE_URL = "http://localhost:8080"
ENDPOINT_REGISTRATION_1 = "/opaque/registration/1"
ENDPOINT_REGISTRATION_2 = "/opaque/registration/2"
ENDPOINT_LOGIN_1 = "/opaque/login/1"
ENDPOINT_LOGIN_2 = "/opaque/login/2"