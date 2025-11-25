import cv2
import time
import numpy as np
from typing import Tuple, Dict
from ultralytics import YOLO
import json
from datetime import datetime
import torch
import os

class PerceptionEngine():
    def __init__(self, model_name, model_threshold):
        self.model = YOLO(model_name)
        self.conf = model_threshold

    # ---------------- Inference Function ----------------
    def get_bounding_boxes(self,image: np.ndarray, conf: float) -> Tuple[np.ndarray, np.ndarray, float]:
        torch.cuda.synchronize()
        start_time = time.time()

        results = self.model.predict(
            source=image,
            device='cuda',  # Run on GPU
            conf=self.conf,
            classes=[0],  # Only 'person' class
            save=False,
            show=False,
            verbose=False
        )

        torch.cuda.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"[INFO] Inference Time: {inference_time:.4f} sec")

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
        return boxes, confs, inference_time

    # ---------------- Drawing Function ----------------
    def draw_bounding_box(self, image: np.ndarray, boxes: np.ndarray, confs: np.ndarray) -> Tuple[np.ndarray, Dict[int, Dict[str, int]]]:
        metadata = {}
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1

            color = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"ID:{i} W:{width}px H:{height}px", (x1, max(0, y1 - 30)), font, font_scale, color, 1)
            cv2.putText(image, f"C:{conf:.2f}", (x1, max(0, y1 - 15)), font, font_scale, color, 1)

            metadata[i] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'width': width, 'height': height}
        return image, metadata