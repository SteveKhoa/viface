from lib.biocryp import keygen
import os

DIRECTORY_CURRENT = os.getenv("DIRECTORY_CURRENT")
USER_ID = os.getenv("CLIENT_TEST_USER_ID")
IMAGE_PATH = os.getenv("CLIENT_IMAGE_PATH")
BIOCRYP_HELPER_DIR = os.getenv("CLIENT_DIRECTORY_BIOCRYP_HELPER")
ENROLL_FROM_STATIC_DATA_DIR = os.getenv("CLIENT_ENROLL_FROM_STATIC_DATA_DIR")
ENROLL_FROM_STATIC_DATA_FLAG = os.getenv("CLIENT_ENROLL_FROM_STATIC_DATA_FLAG")
ENROLL_USER_VAULT_DIR = os.getenv("CLIENT_ENROLL_USER_VAULT_DIR")
TEST_USER_ID = os.getenv("CLIENT_TEST_USER_ID")
MASK_THEN_LOCK_MASK_LENGTH = int(os.getenv("CLIENT_MASK_THEN_LOCK_MASK_LENGTH"))


SERVER_BASE_URL = "http://localhost:8080"
ENDPOINT_REGISTRATION_1 = "/opaque/registration/1"
ENDPOINT_REGISTRATION_2 = "/opaque/registration/2"
ENDPOINT_LOGIN_1 = "/opaque/login/1"
ENDPOINT_LOGIN_2 = "/opaque/login/2"


FEATURE_EXTRACTOR_ENFORCE_DETECTION_FLAG = os.getenv("CLIENT_FEATURE_EXTRACTOR_ENFORCE_DETECTION_FLAG") == "true"


keygen_fuzzy_extractor = keygen.MaskThenLockFuzzyExtractor(
    input_length=512,
    key_length=16,
    mask_length=MASK_THEN_LOCK_MASK_LENGTH,
    nonce_len=1,
)
