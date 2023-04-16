import numpy as np


def speed(previous_positions, current_positions, video_info):
    lane_width = 3.5
    num_of_lane = 3
    object_speeds = {}
    ppm = video_info.width / (lane_width * num_of_lane)
    for tracker_id, current_bbox in current_positions.items():
        previous_bbox = previous_positions.get(tracker_id)
        if previous_bbox is not None:
            dx, dy = current_bbox[:2] - previous_bbox[:2]
            distance_pixel = np.sqrt(dx ** 2 + dy ** 2)
            if distance_pixel > 1:
                distance_meters = distance_pixel / ppm
                speed_kph = "{:.2f}".format(((distance_meters * video_info.fps) / 5) * 3.6)
            else:
                speed_kph = 0
            object_speeds[tracker_id] = float(speed_kph)
    return object_speeds
l