import os
import random
from PIL import Image
from tqdm import tqdm
from utils import parse_annotation, verify_data_split

def preprocess_and_split_data(source_dir, output_dir, class_map, split=0.2):
    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_label_dir = os.path.join(output_dir, "labels", "train")
    val_label_dir = os.path.join(output_dir, "labels", "val")

    for dir in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(dir, exist_ok=True)

    valid_pairs = []
    for f in tqdm(os.listdir(source_dir), desc="Validating files"):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(source_dir, f)
            xml_path = os.path.join(source_dir, f"{os.path.splitext(f)[0]}.xml")
            if os.path.exists(xml_path):
                try:
                    Image.open(img_path).verify()
                    valid_pairs.append((f, xml_path))
                except Exception as e:
                    print(f"Error validating image {f}: {e}")

    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * (1 - split))
    
    process_set(valid_pairs[:split_idx], source_dir, train_img_dir, train_label_dir, class_map, "training")
    process_set(valid_pairs[split_idx:], source_dir, val_img_dir, val_label_dir, class_map, "validation")

    verify_data_split(train_img_dir, train_label_dir)
    verify_data_split(val_img_dir, val_label_dir)

def process_set(pairs, source_dir, img_dir, label_dir, class_map, set_name):
    for img_file, xml_file in tqdm(pairs, desc=f"Processing {set_name} set"):
        try:
            img = Image.open(os.path.join(source_dir, img_file)).convert('RGB')
            new_img_path = os.path.join(img_dir, f"{os.path.splitext(img_file)[0]}.jpg")
            img.save(new_img_path, 'JPEG')

            objects = parse_annotation(xml_file, class_map)
            if objects:
                txt_file = os.path.join(label_dir, f"{os.path.splitext(img_file)[0]}.txt")
                with open(txt_file, 'w') as f:
                    for obj in objects:
                        f.write(' '.join(map(str, obj)) + '\n')
            else:
                os.remove(new_img_path)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
