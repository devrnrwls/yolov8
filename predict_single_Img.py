import cv2
import os
from ultralytics import YOLO

image_path = 'predict_dataset3/19303072-2023-09-18-154708.png'

# OpenCV를 사용하여 이미지 열기
input_img = cv2.imread(image_path)

model = YOLO('runs/detect/train19/weights/last.pt')
results = model.predict(input_img, save=False, conf=0.1)
res_plotted = results[0].plot()

# OpenCV는 이미지를 BGR 색상 순서로 처리하므로 RGB로 변환
# res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

# 이미지 표시
cv2.imshow('YOLO Result', res_plotted)
cv2.waitKey(0)
cv2.destroyAllWindows()
