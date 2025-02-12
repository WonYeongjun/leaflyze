#사진 촬영
import subprocess
import json
with open("config.json", "r") as f:
    config = json.load(f)
    
LOCAL_FILE = config["raspi_file_path"]

subprocess.run(
    [
        "libcamera-jpeg",
        "-o",
        LOCAL_FILE,
        "--width",
        "4608",
        "--height",
        "2592",
        "--shutter",
        "5000",
        "--gain",
        "15",
    ]
)

print("사진 촬영 완료")
