from client import (
    biocryp_image_augmentation,
    biocryp_key_generation,
    deepface_feature_extraction,
)
from protocol import opaque_client_registration
import base64


def execute_registration(cv2_image_imreads, user_id: str):
    images = biocryp_image_augmentation.execute_image_augmentation(cv2_image_imreads)

    feature_vectors = []
    for image in images:
        feature_vector = deepface_feature_extraction.execute_feature_extraction(image)
        feature_vectors += [feature_vector]

    key = biocryp_key_generation.execute_key_generation(feature_vectors, key_id=user_id)
    key_b64 = base64.b64encode(key).decode("ascii")
    response = opaque_client_registration.execute_registration_usecase(user_id, key_b64)

    print(f"Success: {response}")