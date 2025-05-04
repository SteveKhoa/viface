import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from client import login
from client import constant
import cv2

# NOTE: Make sure you run the server first.

if __name__ == "__main__":
    cv2_image_imread = cv2.imread(constant.IMAGE_PATH)

    login.execute_login(cv2_image_imread, constant.USER_ID)
