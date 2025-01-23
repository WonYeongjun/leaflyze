import cv2
import numpy as np

# 캔버스 생성 (흰 배경)
marker_size = 300  # 마커 크기 (픽셀 단위)
marker = np.ones((marker_size, marker_size, 3), dtype=np.uint8) * 255  # 흰색 배경

# "4" 모양 그리기
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 5
font_thickness = 10
text = "4"
text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
text_x = (marker_size - text_size[0]) // 2
text_y = (marker_size + text_size[1]) // 2
cv2.putText(marker, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

# 저장
cv2.imwrite("marker_4.png", marker)

# 결과 보기
cv2.imshow("4 Marker", marker)
cv2.waitKey(0)
cv2.destroyAllWindows()
