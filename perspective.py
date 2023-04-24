import cv2


def transform(img, point1, point2):
    matrix = cv2.getPerspectiveTransform(point1, point2)
    frame = cv2.warpPerspective(img, matrix, (1920, 1080))
    return frame



