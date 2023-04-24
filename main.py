import cv2
import numpy as np
from numpy import ndarray
from typing import List, Union
from onemetric.cv.utils.iou import box_iou_batch
from supervision.draw.color import ColorPalette
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker, STrack
from tqdm import tqdm
from dataclasses import dataclass


from record import record
from estimate_speed import speed
from perspective import transform


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


# Converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# Converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


def match_detections_with_tracks(
        detections: Detections,
        tracks: List[STrack]
) -> Union[ndarray, list[None]]:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


# ============================================= Declare =================================================
CAR_IMG_PATH = ""
CAR_PLATES_PATH = ""
MODEL_PATH = ""
VIDEO_PATH = ""
RESULT_VIDEO_PATH = ""
model = YOLO(MODEL_PATH)
CLASS_NAMES_DICT = model.model.names
CLASS_ID = [2, 3, 5, 7]

# ROI selecting
# a b
# c d
a = []
b = []
c = []
d = []

p1 = np.float32([a, b, c, d])
p2 = np.float32([[0, 0], [1920, 0], [0, 1080], [1920, 1080]])

byte_tracker = BYTETracker(BYTETrackerArgs())
video_info = VideoInfo.from_video_path(VIDEO_PATH)
generator = get_video_frames_generator(VIDEO_PATH)
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=1)
speedLimit = 20
width = video_info.width
height = video_info.height
previous_positions = {}
current_positions = {}
cal_speed = {}
previous_speed = {}
alpha = 0.4
# ======================================================= Main ======================================================
with VideoSink(RESULT_VIDEO_PATH, video_info) as sink:
    for frame_index, frame in enumerate(tqdm(generator, total=video_info.total_frames)):
        origin_frame = frame
        frame = transform(frame, p1, p2)
        print(frame_index)
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )

        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)

        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        tracker_ids = match_detections_with_tracks(detections, tracks)
        current_positions = {
            tracker_id: detections.xyxy[detection_index]
            for detection_index, tracker_id in enumerate(tracker_ids)
            if tracker_id is not None
            for class_id in detections
        }
        if (frame_index % 5) == 0:
            previous_speed = cal_speed
            cal_speed = speed(previous_positions, current_positions, video_info)
            previous_positions = current_positions.copy()
        labels = [
            f"ID{tracker_id} {CLASS_NAMES_DICT[class_id]} {cal_speed.get((tracker_id,) if isinstance(tracker_id, int) else tracker_id) or previous_speed.get(tracker_id)} km\h "
            for _, _, class_id, tracker_id in detections
        ]

        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        roi = transform(frame, p2, p1)
        blended = cv2.addWeighted(origin_frame, alpha, roi, 1 - alpha, 0)

        # ====================================================== Save result ===========================================
        for _, _, class_id, tracker_id in detections:
            try:
                if (cal_speed.get((tracker_id,) if isinstance(tracker_id, int) else tracker_id) or previous_speed.get(
                        tracker_id)) >= speedLimit:
                    print("ok")
                    record(int(tracker_id), cal_speed.get(
                        (tracker_id,) if isinstance(tracker_id, int) else tracker_id) or previous_speed.get(tracker_id),
                              blended, frame, current_positions, CAR_IMG_PATH, CAR_PLATES_PATH)
            except TypeError:
                print("None speed")
        sink.write_frame(blended)

        # ====================================================== Display frame =========================================
        roi_vertices = np.array([[p1[0], p1[1], p1[3], p1[2]]], dtype=np.int32)
        cv2.polylines(origin_frame, roi_vertices, True, (0, 0, 255), 3)
        cv2.imshow("Frame", origin_frame)
        cv2.imshow("ROI", roi)
        cv2.imshow("", frame)
        cv2.imshow("Result", blended)
        cv2.waitKey(1)
cv2.destroyAllWindows()
