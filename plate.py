import cv2
import numpy as np
from easyocr import Reader


def plate_detection(image, region):
    """""
    region:
        Thailand: th
        English: en
        Vietnam: vn
    """
    image_np = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    n_plate_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

    detections = n_plate_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

    number_plate = None
    for (x, y, w, h) in detections:
        number_plate = gray[y:y + h, x:x + w]
    if number_plate is None:
        return "Not Found!"

    reader = Reader([region])
    detection = reader.readtext(number_plate)
    if len(detection) == 0:
        return "Not Found!"
    else:
        return f"{detection[0][1]}"
