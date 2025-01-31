#십자 이미지에서 픽셀-실제거리 사이 계수 도출
import cv2
import numpy as np
import math
from picamera2 import Picamera2

def calculate_cross_distance(frame, real_distance_mm):
    """프레임에서 십자 중심 좌표 및 픽셀-실제 거리 변환 계수를 계산"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)  # 이진화 (십자가가 검정색일 경우)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []  # 십자가 중심 좌표 리스트
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
            # 중심점 표시
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    if len(centers) >= 2:
        # 두 십자 중심 좌표 계산
        p1, p2 = centers[:2]
        pixel_distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        conversion_factor = real_distance_mm / pixel_distance  # 픽셀-실제 거리 계수
        return centers, pixel_distance, conversion_factor
    return centers, None, None

def camera_stream_and_calculate(real_distance_mm):
    """실시간 카메라 스트리밍과 십자 계산"""
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration())
    picam2.start()

    print("실시간 스트리밍 중입니다. 'q' 키를 눌러 계산 결과를 확인하세요.")
    while True:
        frame = picam2.capture_array()  # 프레임 캡처
        display_frame = frame.copy()  # 화면에 표시할 프레임 복사본

        # 십자 탐지 및 거리 계산
        centers, pixel_distance, conversion_factor = calculate_cross_distance(display_frame, real_distance_mm)

        # 화면에 정보 표시
        if len(centers) >= 2 and pixel_distance is not None:
            cv2.putText(display_frame, f"Pixel Distance: {pixel_distance:.2f}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Conversion: {conversion_factor:.4f} mm/px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Detecting crosses...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 실시간 화면 출력
        cv2.imshow("Camera Stream", display_frame)

        # 'q' 키를 누르면 스트리밍 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("계산을 종료합니다.")
            break

    # 결과 출력
    if len(centers) >= 2 and pixel_distance is not None:
        print(f"첫 번째 중심 좌표: {centers[0]}")
        print(f"두 번째 중심 좌표: {centers[1]}")
        print(f"픽셀 거리: {pixel_distance:.2f} px")
        print(f"실제 거리: {real_distance_mm} mm")
        print(f"픽셀-실제 거리 변환 계수: {conversion_factor:.4f} mm/px")
    else:
        print("십자 표시가 감지되지 않았습니다.")

    # 자원 해제
    picam2.close()
    cv2.destroyAllWindows()

# 실행
real_distance_mm = 61.5  # 십자 중심 간 실제 거리(mm)
camera_stream_and_calculate(real_distance_mm)
