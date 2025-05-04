from client import (
    biocryp_key_reproduction,
    deepface_feature_extraction,
)
from protocol import opaque_client_login
import base64
import cv2


def execute_login(cv2_image_imread, user_id: str):
    face_image = cv2.cvtColor(cv2_image_imread, cv2.COLOR_BGR2RGB)
    feature_vector = deepface_feature_extraction.execute_feature_extraction(face_image)

    key = biocryp_key_reproduction.execute_key_reproduction(feature_vector, key_id=user_id)
    key_b64 = base64.b64encode(key).decode("ascii")
    response = opaque_client_login.execute_login_usecase(user_id, key_b64)

    print(f"Success: {response}")