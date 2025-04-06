import subprocess
from pathlib import Path
from IPython.display import Video
from ultralytics import YOLO
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def run_video_detection():
    model_path = config["test"]["model_path"]
    video_path = Path(config["test"]["video_path"])
    yolo = YOLO(model_path)

    results_video = yolo.predict(
        video_path,
        conf=0.5,
        save=True,
        stream=True,
        project=Path("runs", "detect"),
        name="video_source"
    )

    _ = [result for result in results_video]  # Process video frames

    # Dynamically get .avi filename
    input_video = Path("runs/detect/video_source", video_path.name).with_suffix(".avi")
    output_video = input_video.with_suffix(".mp4")

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_video), "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-c:a", "aac", "-b:a", "128k", str(output_video)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print("Video conversion complete:", output_video.name)
    return Video(str(output_video))

if __name__ == "__main__":
    run_video_detection()