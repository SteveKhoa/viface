"""Feature Extractor Module
"""

from deepface import DeepFace  # unifying the interface of Feature Extractor functions
from PIL.Image import Image
import numpy as np
from typing import Union, Tuple
from sklearn.preprocessing import normalize


def extract_feature_vector(
    image: Union[Image, np.ndarray],
    model_name: str = "GhostFaceNet",
    face_detector_backend: str = "ssd",
    enforce_detection: bool = False,
) -> Tuple[np.ndarray, bool]:
    npimg = _convert_to_np(image)

    try:
        embedding = DeepFace.represent(
            img_path=npimg,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend=face_detector_backend,
            max_faces=1,
        )[0]["embedding"]
    except ValueError:
        # No face detected

        return (None, False)

    embedding = np.array(embedding)
    embedding = np.expand_dims(embedding, axis=0)
    embedding = normalize(embedding)
    embedding = np.squeeze(embedding)

    return (np.array(embedding), True)


def detect_face(
    image: Union[Image, np.ndarray],
    model_name: str = "DeepID",
    face_detector_backend: str = "ssd",
):
    npimg = _convert_to_np(image)

    try:
        facial_area = DeepFace.represent(
            img_path=npimg,
            model_name=model_name,
            enforce_detection=True,
            detector_backend=face_detector_backend,
            max_faces=1,
        )[0]["facial_area"]

        x = facial_area["x"]
        y = facial_area["y"]
        w = facial_area["w"]
        h = facial_area["h"]

        ok = True
    except ValueError:
        # Face cannot be detected

        x = None
        y = None
        w = None
        h = None
        
        ok = False

    return (x, y, w, h, ok)


def _convert_to_np(image: Union[Image, np.ndarray]):
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

    return npimg
