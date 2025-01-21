import subprocess

# 사진 촬영 (1920x1440 해상도)
subprocess.run(["libcamera-jpeg", "-o", "/home/userk/cal_img/raw/raw_img.jpg", "--width", "1920", "--height", "1440"])

print("사진 촬영 완료")
