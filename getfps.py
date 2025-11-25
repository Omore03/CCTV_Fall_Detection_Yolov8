import cv2
import glob
import os

clips_dir = "clips"
for path in sorted(glob.glob(os.path.join(clips_dir, "*.mp4"))):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{os.path.basename(path)}: {fps} FPS")
    cap.release()
