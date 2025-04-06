from ultralytics import YOLO
import os

def train_model(data_yaml_path, config):
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

    model = YOLO("yolov8n.pt")  # or use config path

    print("Starting training with config:", config)

    try:
        training_args = {
            "data": data_yaml_path,
            "epochs": config.get("epochs", 50),
            "batch": config.get("batch", 8),
            "device": config.get("device", "cuda"),
            "patience": config.get("patience", 5),
            "workers": config.get("workers", 2),
        }

        # If augmentations are enabled, add augmentation-specific parameters
        if config.get("augmentations", False):
            training_args.update({
                "augment": True,
                "mosaic": config.get("mosaic", 1.0),
                "mixup": config.get("mixup", 0.1),
                "degrees": config.get("degrees", 10.0),
                "translate": config.get("translate", 0.1),
                "scale": config.get("scale", 0.5),
                "shear": config.get("shear", 2.0),
                "perspective": config.get("perspective", 0.0),
                "flipud": config.get("flipud", 0.5),
                "fliplr": config.get("fliplr", 0.5),
                "hsv_h": config.get("hsv_h", 0.015),
                "hsv_s": config.get("hsv_s", 0.7),
                "hsv_v": config.get("hsv_v", 0.4),
            })

        results = model.train(**training_args)
        return results

    except Exception as e:
        print(f"Error during training: {e}")
        raise
