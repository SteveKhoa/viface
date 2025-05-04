import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from deepface import DeepFace  # unifying the interface of Feature Extractor functions
from PIL.Image import Image
import numpy as np
from typing import Union
import sklearn.preprocessing
import time
import cv2


IMAGE_PATH = os.getenv("IMAGE_PATH")

# Number of times the experiment is re-run
N_RETRY = 10

MODEL_NAMES = ["Facenet512", "Facenet", "VGG-Face", "ArcFace", "GhostFaceNet"]


def benchmark_execute_feature_extraction(
    image: Union[Image, np.ndarray], model_names
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

    for model_name in model_names:
        start = time.time()

        embedding = DeepFace.represent(
            img_path=npimg,
            model_name=model_name,
            enforce_detection=True,
            detector_backend="ssd",
            max_faces=1,
        )[0]["embedding"]

        end = time.time()

        print(
            f"model={model_name}, shape={np.array(embedding).shape}, elapsed={end - start} "
        )

    embedding = np.array(embedding)
    embedding = np.expand_dims(embedding, axis=0)
    embedding = sklearn.preprocessing.normalize(embedding)
    embedding = np.squeeze(embedding)

    return np.array(embedding)


if __name__ == "__main__":
    cv2_image_imread = cv2.imread(IMAGE_PATH)

    for i in range(N_RETRY):
        benchmark_execute_feature_extraction(cv2_image_imread, model_names=MODEL_NAMES)
