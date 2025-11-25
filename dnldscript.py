import urllib.request

models = {
    "YOLOv8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
    "YOLOv8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
    "YOLOv8l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
    "YOLOv8x.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
}

for filename, url in models.items():
    print(f"Downloading {filename} ...")
    urllib.request.urlretrieve(url, filename)
    print(f"Saved: {filename}")

print("\nAll YOLOv8 models downloaded.")
