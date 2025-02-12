# 버튼 누르면 사진 찍고 pc로 전송송
import RPi.GPIO as GPIO
import time
import subprocess
import threading
import paramiko
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

BUTTON_PIN = 17

HOST = config["ip_address"]
PORT = 22  # SSH 포트 (기본: 22)
USERNAME = config["username"]
PASSWORD = config["passward"]
LOCAL_FILE = config["raspi_file_path"]
REMOTE_PATH = config["pc_file_path"]

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


def capture_photo():
    """사진을 촬영하고 전송 및 실행"""
    print("📸 사진 촬영 중...")
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
    print("사진 촬영 완료!")

    send_file_to_windows()


def send_file_to_windows():
    """SSH(SFTP)를 사용하여 Windows로 파일 전송 및 실행"""
    try:
        print("📂 파일 전송 중...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(HOST, PORT, USERNAME, PASSWORD)

        sftp = ssh.open_sftp()
        sftp.put(LOCAL_FILE, REMOTE_PATH)
        sftp.close()

        print("✅ 파일 전송 완료!")

        ssh.close()

    except Exception as e:
        print(f"❌ 오류 발생: {e}")


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
