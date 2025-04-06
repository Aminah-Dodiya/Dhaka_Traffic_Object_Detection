import os
import torch
import yaml
from src.preprocess import preprocess_and_split_data
from src.train import train_model
from src.utils import get_classes, create_data_yaml, verify_directory_structure
from utils import load_config

def main():
    # Load configuration
    config_path = os.path.join("config", "config.yaml")
    config = load_config(config_path)

    # Choose training config: 'train' or 'train_augmented'
    training_config = config["train_augmented"]  # ← switch to "train" if you want baseline

    source_dir = config["train"]["source_dir"]
    output_dir = os.path.abspath(config["train"]["output_dir"])
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    device = torch.device(config["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    epochs = config["train"].get("epochs", 30)
    batch_size = config["train"].get("batch_size", 8)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Source directory: {source_dir}")
    print(f"[INFO] Output directory: {output_dir}")

    # Step 1: Get class mapping
    xml_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.xml')]
    class_map = get_classes(xml_files)

    # Step 2: Preprocess and split data
    preprocess_and_split_data(source_dir, output_dir, class_map)

    # Step 3: Create data.yaml for YOLOv8
    create_data_yaml(output_dir, class_map, data_yaml_path)

    # Step 4: Verify directory structure
    verify_directory_structure(output_dir)

    # Step 5: Train the model
    train_model(data_yaml_path, training_config)

if __name__ == "__main__":
    main()
