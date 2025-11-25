import cv2
import os
import datetime

def get_video_general_information(video_path):
    info = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    # Basic video properties
    info['fps'] = cap.get(cv2.CAP_PROP_FPS)
    info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info['duration_seconds'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else None
    info['duration_hms'] = str(datetime.timedelta(seconds=int(info['duration_seconds']))) if info['duration_seconds'] else None
    info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Codec info
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    info['codec'] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    return info


# ---------------- State Machine Helpers ----------------
def is_width_greater_than_height(metadata: dict) -> bool:
    return any(coord['width'] > coord['height'] for coord in metadata.values())

def is_height_greater_than_width(metadata: dict) -> bool:
    return any(coord['height'] > coord['width'] for coord in metadata.values())