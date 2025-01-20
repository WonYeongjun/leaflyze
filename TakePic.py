from picamzero import Camera

# 카메라 초기화
cam = Camera()



# 카메라 프리뷰 시작
cam.start_preview()

# 사진 촬영 및 저장
cam.take_photo("/home/userk/cal_img/raw/raw_img.jpg")  # 확장자는 .jpg 또는 .png로 지정

# 카메라 프리뷰 중지
cam.stop_preview()
