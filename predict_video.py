import cv2
from ultralytics import YOLO

# AVI 파일 경로
video_path = 'test.mp4'

# VideoCapture 객체 생성
cap = cv2.VideoCapture(video_path)

# YOLO 모델 초기화
model = YOLO('runs/detect/train19/weights/last.pt')

# 비디오 결과를 저장하기 위한 VideoWriter 객체 설정
output_path = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
out = cv2.VideoWriter(output_path, fourcc, 30.0, (2048, 1536))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 여러 객체를 컨투어로 구분해서 잡아줌
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y + h, x:x + w]

        if w > 200 and h > 200:

            roi = frame[y:y + h, x:x + w]
            results = model.predict(roi, save=False, conf=0.6)

            #plot explanination: https://docs.ultralytics.com/modes/predict/#plotting-results
            res_plotted = results[0].plot()

            frame[y:y + h, x:x + w] = res_plotted

            # cv2.imshow('Object Detection', resized_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    #영상 저장 사이즈와 이미지 사이즈가 다르면 저장이 안됨
    # resized_image = cv2.resize(frame, (640, 480))
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
