import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))

import cv2
import time
from lib.feature import dnn


def capture_to_cv2_multiple_images():
    """
    Upon triggered, OpenCV webcam API is called to acquire a sequence of images.
    The capture only stops when user press key "Enter".

    This function then returns that sequences of images via read.
    """

    images = []
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture - Press Enter to stop", frame)
        if cv2.waitKey(1) == 13:  # Enter key
            break
        images.append(frame.copy())
    cap.release()
    cv2.destroyAllWindows()
    return images


def capture_to_cv2_single_image():
    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        x, y, w, h, ok = dnn.detect_face(frame)

        if ok:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "OK", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Capture - Press Enter to stop", frame)

        if cv2.waitKey(100) == 13:  # Enter key
            captured_image = frame
            break

        time.sleep(0.5)

    cap.release()

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return captured_image


if __name__ == "__main__":
    capture_to_cv2_single_image()
    time.sleep(2)
