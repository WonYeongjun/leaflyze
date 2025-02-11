#사진 촬영
import subprocess
LOCAL_FILE = "/home/userk/cal_img/raw/raw_img.jpg"#사진 저장 위치

# 사진 촬영
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
