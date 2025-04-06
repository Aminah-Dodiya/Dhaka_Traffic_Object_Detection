import os
import xml.etree.ElementTree as ET
import yaml
from tqdm import tqdm

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_classes(xml_files):
    classes = set()
    for xml_file in tqdm(xml_files, desc="Extracting classes"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name:
                    classes.add(class_name)
        except ET.ParseError as e:
            print(f"Error parsing {xml_file}: {e}")
        except Exception as e:
            print(f"Unexpected error with {xml_file}: {e}")
    
    return {cls: idx for idx, cls in enumerate(sorted(classes))}

def parse_annotation(xml_file, class_map):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in class_map:
            continue
        
        cls_id = class_map[cls_name]
        bbox = obj.find("bndbox")
        xmin, ymin, xmax, ymax = [float(bbox.find(coord).text) for coord in ["xmin", "ymin", "xmax", "ymax"]]
        
        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        
        objects.append([cls_id, x_center, y_center, box_width, box_height])
    
    return objects

def verify_data_split(image_dir, label_dir):
    image_files = set(os.path.splitext(f)[0] for f in os.listdir(image_dir))
    label_files = set(os.path.splitext(f)[0] for f in os.listdir(label_dir))

    missing_labels = image_files - label_files
    missing_images = label_files - image_files

    if missing_labels:
        print(f"Warning: Missing labels for {len(missing_labels)} images.")
    if missing_images:
        print(f"Warning: Missing images for {len(missing_images)} labels.")

def create_data_yaml(output_dir, class_map, yaml_path):
    data = {
        'train': os.path.abspath(os.path.join(output_dir, 'images', 'train')),
        'val': os.path.abspath(os.path.join(output_dir, 'images', 'val')),
        'nc': len(class_map),
        'names': list(class_map.keys())
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)

def verify_directory_structure(output_dir):
    required_dirs = [
        os.path.join(output_dir, 'images', 'train'),
        os.path.join(output_dir, 'images', 'val'),
        os.path.join(output_dir, 'labels', 'train'),
        os.path.join(output_dir, 'labels', 'val')
    ]
    for dir in required_dirs:
        if os.path.exists(dir):
            print(f"Directory exists: {dir}")
            print(f"Contents: {os.listdir(dir)}")
        else:
            print(f"Directory not found: {dir}")
