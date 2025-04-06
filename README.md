# AI-Based Dhaka Traffic Detection

---

## Overview
Urban traffic detection is a critical component of intelligent transportation systems, especially in cities like **Dhaka**, Bangladesh, where the road infrastructure is limited to just 7% of the city’s area. With over **8 million vehicles moving daily** in a compact 306 sq km space, Dhaka presents a uniquely dense and heterogeneous traffic scenario. Traditional rule-based systems struggle to scale in such environments due to occlusion, vehicle diversity, and inconsistent lighting or angles.

This project addresses this challenge by applying **YOLOv8**, a state-of-the-art real-time object detection model, to develop a robust solution for **automatic multi-class vehicle detection** in Dhaka traffic scenes. It leverages both image and video data, performs **custom training**, and includes **augmentation-based improvements** to increase generalizability.

The goal is not only to build a scalable detection pipeline but also to serve as a foundation for **smart city** traffic management and AI-powered civic infrastructure. Furthermore, this initiative aims to engage regional researchers and developers in South-East Asia to build an AI-driven community around real-world urban challenges.

![Object Detection GIF](image/dhaka_traffic_object_detection.gif)

---

## Data

### Image Dataset
- The dataset contains annotated traffic images captured in Dhaka, each containing one or more vehicles from the following **21 classes**:
  - *ambulance, auto-rickshaw, bicycle, bus, car, garbage van, human hauler, minibus, minivan, motorbike, pickup, army vehicle, police car, rickshaw, scooter, SUV, taxi, three-wheelers (CNG), truck, van, wheelbarrow*
- Annotation Format: Pascal VOC (XML)
- Source: [Harvard Dataverse Link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/POREXF)

### Video Dataset
-A real-world video sourced from **YouTube** is used to test detection on live traffic scenes:
- Source: [YouTube - Dhaka Traffic](https://www.youtube.com/watch?v=0B2-cR4GEjc&list=WL&index=72&t=22s)

---

## Project Structure
```bash
Dhaka_Traffic_Object_Detection
├── config/
│   └── config.yaml
├── image/
│   ├── dhaka_traffic_object_detection.gif
├── models/
│   ├── yolov8_baseline.pt
│   ├── yolov8_custom.pt
│   └── yolov8_custom_aug.pt
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── results_visualization.ipynb
├── scripts/
│   ├── main.py
│   └── test_images.py
│   ├── test_video.py
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── utils.py
└── README.md
├── requirements.txt
```
---

## Model Architecture

- Backbone: `YOLOv8n` (Pretrained on COCO)
- Head: Detection layer for 21 custom classes
- Formats: Trained and saved in PyTorch `.pt` files

We trained three model variants:
1. **Baseline** — Pretrained weights, no fine-tuning
2. **Custom** — Fine-tuned on Dhaka dataset
3. **Augmented** — Fine-tuned with heavy augmentations (mosaic, mixup, HSV, shear, etc.)

---

## Dependencies

Install required packages with:

```bash
pip install -r requirements.txt
```
---

## Usage

### Train the model
```bash
python scripts/main.py
```
-Set training parameters in config/config.yaml:
train: for baseline
train_augmented: for augmented training

### Run detection on test images
```bash
python scripts/test_images.py
```

### Run detection on video
```bash
python scripts/test_video.py
```
The output video will be saved and converted to .mp4 format using ffmpeg.

---

## Results
Key findings from the analysis:
-Baseline Model: Performs well on common classes like cars and buses but struggles with rare classes like ambulances and police cars.
-Custom Model: Improved detection accuracy for Dhaka-specific vehicle types such as rickshaws and three-wheelers (CNG).
-Augmented Model: Enhanced background discrimination and recall for rare classes but slightly reduced overall mAP compared to the non-augmented model.
-Detailed results are available in result_visualization.ipynb.

---

## Future Work
-Build an end-to-end real-time web app for Dhaka traffic monitoring
-Expand dataset to nighttime and rainy conditions
-Use larger YOLOv8 variants (e.g., v8m, v8l)
-Enhance recall for rare classes through advanced augmentation techniques.
-Address persistent misclassifications between visually similar classes (e.g., SUVs vs taxis).

---

## License
This project is open-sourced under the MIT License. Please refer to the LICENSE file for more information.