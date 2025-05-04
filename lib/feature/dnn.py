"""Feature Extractor Module
"""

from deepface import DeepFace  # unifying the interface of Feature Extractor functions
from PIL.Image import Image
import numpy as np
from typing import Union
from sklearn.preprocessing import normalize


def extract_feature_vector(
    image: Union[Image, np.ndarray],
    model_name: str = "GhostFaceNet",
    face_detector_backend: str = "ssd",
) -> np.ndarray:
    if isinstance(image, Image):
        # Convert to numpy array for compatibility
        npimg = (
            np.array(image.convert("RGB").getdata())
            .reshape(
                image.size[0],
                image.size[1],
                3,
            )
            .astype("uint8")
        )
    else:
        npimg = image

    embedding = DeepFace.represent(
        img_path=npimg,
        model_name=model_name,
        enforce_detection=False,
        detector_backend=face_detector_backend,
        max_faces=1,
    )[0]["embedding"]

    embedding = np.array(embedding)
    embedding = np.expand_dims(embedding, axis=0)
    embedding = normalize(embedding)
    embedding = np.squeeze(embedding)

    return np.array(embedding)
