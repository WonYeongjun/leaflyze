import paramiko

# 전송할 파일 정보
local_file = "/home/userk/cal_img/raw/raw_img.jpg"  # 라즈베리파이에 저장된 이미지 파일
remote_path = "C:/Users/UserK/Desktop/raw/raw_img.jpg"  # 윈도우즈 PC에 저장할 경로

# 윈도우즈 PC의 SSH 정보 (OpenSSH 서버 활성화 필요)
host = "172.30.1.66"  # 윈도우즈 PC의 IP 주소
port = 22  # 기본 SSH 포트
username = "USERK"  # 윈도우즈 사용자 계정
password = "1234"  # 계정 비밀번호

# SSH 클라이언트 및 SFTP 세션 생성
try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)

    sftp = ssh.open_sftp()
    sftp.put(local_file, remote_path)  # 파일 전송
    sftp.close()

    ssh.close()
    print("파일 전송 완료!")

except Exception as e:
    print(f"파일 전송 중 오류 발생: {e}")
