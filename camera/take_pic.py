#사진 촬영
import subprocess
LOCAL_FILE = "/home/userk/cal_img/raw/raw_img.jpg"

# 사진 촬영 (노출, 감도 조절 추가)
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
