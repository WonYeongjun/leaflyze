import subprocess

# 사진 촬영 (노출, 감도 조절 추가)
subprocess.run([
    "libcamera-jpeg", "-o", "/home/userk/cal_img/raw/raw_img.jpg",
    "--width", "7680", "--height", "5760",
    "--shutter", "5000",  # 셔터 속도 (마이크로초 단위, 값이 작을수록 어두워짐)
    "--gain", "1"         # 감도 (ISO와 유사, 값이 작을수록 어두워짐)
])

print("사진 촬영 완료")
