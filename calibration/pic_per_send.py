#2초마다 한번 사진촬영 후 원근감 보정, 전송 총 10장
import subprocess
import time
import os
import cv2
import numpy as np
import paramiko

# 디렉토리 설정
raw_dir = "/home/userk/cal_img/raw"
fin_dir = "/home/userk/cal_img/fin"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(fin_dir, exist_ok=True)

# SSH 설정 (Windows PC로 파일 전송)
host = "172.30.1.66"
port = 22
username = "USERK"
password = "1234"
remote_dir = "C:/Users/UserK/Desktop/fin/"

# 원근 보정 함수
def correct_perspective(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return False
    
    h, w, _ = image.shape
    image_center = np.array([w / 2, h / 2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, _ = detector.detectMarkers(gray)
    desired_ids = [12, 18, 27, 5]
    
    if ids is not None:
        marker_points = []
        for marker_id in desired_ids:
            for i, detected_id in enumerate(ids):
                if detected_id[0] == marker_id:
                    marker_corners = corners[i][0]
                    closest_corner = min(marker_corners, key=lambda pt: np.linalg.norm(pt - image_center))
                    marker_points.append(closest_corner)
                    break
        
        if len(marker_points) == 4:
            pts1 = np.array(marker_points, dtype="float32")
            width, height = 4500, 3500
            pts2 = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
            matrix, _ = cv2.findHomography(pts1, pts2)
            dst = cv2.warpPerspective(image, matrix, (width, height))
            cv2.imwrite(output_path, dst)
            print(f"Corrected image saved to {output_path}")
            return True
        else:
            print("Not enough markers detected.")
    else:
        print("No ArUco markers detected.")
    
    return False

# 파일 전송 함수
def send_file(local_file, remote_file):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, port, username, password)
        sftp = ssh.open_sftp()
        sftp.put(local_file, remote_file)
        sftp.close()
        ssh.close()
        print(f"파일 전송 완료: {remote_file}")
    except Exception as e:
        print(f"파일 전송 중 오류 발생: {e}")

# 10장 촬영 & 처리
for i in range(1, 11):
    raw_path = f"{raw_dir}/raw_img_{i}.jpg"
    fin_path = f"{fin_dir}/fin_img_{i}.jpg"
    
    # 사진 촬영
    subprocess.run([
        "libcamera-jpeg", "-o", raw_path,
        "--width", "7680", "--height", "5760",
        "--shutter", "5000",
        "--gain", "1"
    ])
    print(f"사진 촬영 완료: {raw_path}")

    # 원근 보정
    if correct_perspective(raw_path, fin_path):
        # 파일 전송 (보정된 이미지)
        send_file(fin_path, remote_dir + f"fin_img_{i}.jpg")
    
    if i < 10:
        time.sleep(2)  # 2초 대기
