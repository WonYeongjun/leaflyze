import RPi.GPIO as GPIO
import time
import subprocess
import threading
import paramiko
import os  # 파일명 추출을 위해 추가

BUTTON_PIN = 17  # GPIO 17번 핀 (물리적 번호 11번)

# Windows PC의 SSH 정보
HOST = "192.168.0.2"  # Windows PC의 IP 주소
#HOST = "172.30.1.42" # 영준 IP
PORT = 22  # SSH 포트 (기본: 22)
USERNAME = "USERK"  # Windows 계정 이름
PASSWORD = "1234"  # Windows 비밀번호 (보안상 SSH 키 인증 권장)

# 파일 경로 설정
LOCAL_FILE = "/home/userk/cal_img/raw/raw_img.jpg"  # 라즈베리파이의 촬영 파일 위치
REMOTE_PATH = "C:/Users/UserK/Documents/GitHub/leaflyze/exm/glass/white"  # Windows 저장 디렉토리

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


def capture_photo():
    """사진을 촬영하고 Windows로 전송 및 실행"""
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
    print("✅ 사진 촬영 완료!")

    # Windows로 파일 전송 및 실행
    send_file_to_windows()


def send_file_to_windows():
    """SSH(SFTP)를 사용하여 Windows로 파일 전송 및 실행"""
    try:
        print("📂 파일 전송 중...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(HOST, PORT, USERNAME, PASSWORD)

        sftp = ssh.open_sftp()
        
        # 원격 저장 경로에 시간정보를 포함한 파일명 생성
        original_file_name = os.path.basename(LOCAL_FILE)         # 예: raw_img.jpg
        name, ext = os.path.splitext(original_file_name)           # 분리: ("raw_img", ".jpg")
        timestamp = time.strftime("%Y%m%d_%H%M%S")                   # 예: 20250207_153045
        new_file_name = f"{name}_{timestamp}{ext}"                  # 예: raw_img_20250207_153045.jpg
        
        # REMOTE_PATH가 디렉토리 경로일 경우, 파일명과 결합하여 전체 경로 생성
        remote_file_full_path = REMOTE_PATH.rstrip('/') + '/' + new_file_name
        
        # 파일 전송 (원격 경로에 시간정보가 포함된 파일명으로 저장됨)
        sftp.put(LOCAL_FILE, remote_file_full_path)
        sftp.close()

        print(f"✅ 파일 전송 완료! 저장 위치: {remote_file_full_path}")

        # Windows에서 perspective_win.py 실행 (run_script.bat이 해당 스크립트를 호출한다고 가정)
        command = 'cmd /c "C:/Users/UserK/Desktop/run_script.bat"'
        stdin, stdout, stderr = ssh.exec_command(command)

        print("실행 시작")

        # 입력이 필요한 경우 자동 입력 (필요한 경우 수정)
        stdin.write("your_input_value\n")
        stdin.flush()

        # 실행 결과 실시간 출력
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                print(stdout.readline().strip())
            if stderr.channel.recv_ready():
                print(stderr.readline().strip())

        ssh.close()
        print("🚀 Windows에서 perspective_win.py 실행 완료!")

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
