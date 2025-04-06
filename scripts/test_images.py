import os
from pathlib import Path
from ultralytics import YOLO
import yaml
from PIL import Image

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def run_image_inference():
    model_path = config["test"]["model_path"]
    test_dir = Path(config["test"]["test_images_path"])
    output_dir = Path("runs", "predict", "test_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    for file in os.listdir(test_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = test_dir / file
            results = model.predict(source=img_path, save=True, save_txt=True, conf=0.5)
            print(f"Predicted: {file}")

if __name__ == "__main__":
    run_image_inference()
