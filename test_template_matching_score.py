import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지와 템플릿 로드
# image = cv2.imread("image.jpg", 0)
# template = cv2.imread("template.jpg", 0)
image = cv2.imread("cloth4.jpg", 0)
template = plt.imread("marker44.jpg", 0)
from IPython import embed

embed()
exit()
# 템플릿 매칭
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# 결과를 0과 1 사이로 정규화
normalized_result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)

# 시각화
plt.imshow(normalized_result, cmap="hot")
plt.colorbar()
plt.title("Template Matching Score")
plt.show()
