import dlib
import cv2
import numpy as np
from tensorflow.keras.models import load_model

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

# 얼굴 형 분류를 위한 미리 학습된 CNN 모델 로드
face_shape_model = load_model("face-shape-recognizer.h5")

# 이미지 파일 경로
image_file = "image3.png"

print("Processing file: {}".format(image_file))
img = dlib.load_rgb_image(image_file)

# set number of upsampling according to your test image
dets = detector(img, 2)
print("    - Number of faces detected: {}".format(len(dets)))
for index, det in enumerate(dets):
    print("        - Detection: {}: LEFT: {}, TOP: {}, RIGHT: {}, BOTTOM: {}".format(
        index, det.left(), det.top(), det.right(), det.bottom()))

    # 얼굴 영역 추출
    face_roi = img[det.top():det.bottom(), det.left():det.right()]

    # 얼굴 형 분류를 위한 입력 전처리
    face_roi = cv2.resize(face_roi, (190, 250))  # 크기 조정
    face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_roi_gray = face_roi_gray / 255.0  # 이미지 정규화
    face_roi_gray = np.expand_dims(face_roi_gray, axis=-1)  # 차원 확장

    # 얼굴 형 분류 실행
    predictions = face_shape_model.predict(np.expand_dims(face_roi_gray, axis=0))
    face_shape_label = np.argmax(predictions)

    # 예측 결과 확인
    face_shape_name = ["Oval", "Round", "Square", "Heart", "Oblong"][face_shape_label]

    # 얼굴 영역과 얼굴 형 표시
    cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (0, 255, 0), 2)
    cv2.putText(img, face_shape_name, (det.left(), det.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

win.clear_overlay()
win.set_image(img)
win.add_overlay(dets)
dlib.hit_enter_to_continue()
