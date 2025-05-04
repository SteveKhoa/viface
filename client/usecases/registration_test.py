import sys
import os

from client.usecases import registration

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from client import constant
from os import urandom
import cv2

# NOTE: Make sure you run the server first.

if __name__ == "__main__":
    cv2_image_imread = cv2.imread(constant.IMAGE_PATH)

    registration.execute_registration(cv2_image_imread, constant.USER_ID)
