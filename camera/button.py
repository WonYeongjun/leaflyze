import RPi.GPIO as GPIO
import time
import subprocess
import threading
import paramiko

BUTTON_PIN = 17  # GPIO 17 (핀 번호 11)

# Windows PC의 SSH 정보
HOST = "192.168.143.60"  # Windows PC의 IP
PORT = 22  # SSH 포트
USERNAME = "USERK"  # Windows 계정
PASSWORD = "1234"  # Windows 비밀번호

# 파일 경로 설정
LOCAL_FILE = "/home/userk/cal_img/raw/raw_img.jpg"  # 라즈베리파이의 촬영 파일
REMOTE_PATH = "C:/Users/UserK/Desktop/raw/raw_img.jpg"  # Windows 저장 경로

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def capture_photo():
    """사진을 촬영하고 Windows로 전송"""
    print("📸 사진 촬영 중...")
    subprocess.run([
    "libcamera-jpeg", "-o", "marker.jpg",
    "--width", "4608", "--height", "2592",
    "--shutter", "3000", "--gain", "15",
    "--ev", "-1", "--contrast", "2.0",
    "--brightness", "-0.1", "--sharpness", "2.0",
    "--awb", "greyworld"
    ])
    print("✅ 사진 촬영 완료!")

    # Windows로 파일 전송
    send_file_to_windows()

def send_file_to_windows():
    """SSH(SFTP)를 사용하여 Windows로 파일 전송"""
    try:
        print("📂 파일 전송 중...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(HOST, PORT, USERNAME, PASSWORD)

        sftp = ssh.open_sftp()
        sftp.put(LOCAL_FILE, REMOTE_PATH)
        sftp.close()
        ssh.close()

        print("✅ 파일 전송 완료!")
    except Exception as e:
        print(f"❌ 파일 전송 중 오류 발생: {e}")

def button_callback(channel):
    """버튼이 눌리면 실행"""
    print("🔘 버튼이 눌렸습니다!")
    threading.Thread(target=capture_photo, daemon=True).start()

GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING, callback=button_callback, bouncetime=200)

try:
    print("🔴 버튼을 눌러보세요 (종료: Ctrl+C)")
    while True:
        time.sleep(0.1)

except KeyboardInterrupt:
    print("🛑 프로그램 종료")
    GPIO.cleanup()
