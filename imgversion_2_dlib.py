import cv2
import dlib
import numpy as np
import os
import time
from keras.models import load_model

# 얼굴 탐지기 초기화
detector = dlib.get_frontal_face_detector()

# 내장 카메라 실행
cap = cv2.VideoCapture(0)

# 카메라 프레임 크기 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # 현재 프레임 가져오기
    ret, frame = cap.read()

    # 프레임 가져오기 실패 시 종료
    if not ret:
        break

    # 이미지 출력
    cv2.imshow('Camera', frame)

    # 키 입력 대기 (25ms)
    key = cv2.waitKey(25)
#------------------------------------------------------------- 표정인식
    # 'c' 키 누르면 현재 프레임 캡처
    if key == ord('r'):
        # 현재 시간을 파일 이름으로 사용하여 캡처 저장
        filename = str(time.time()) + '.jpg'
        cv2.imwrite(filename, frame)
        print(f'{filename} saved!')

        # 저장한 파일을 이용하여 얼굴 탐지 및 감정 예측
        image_path = filename
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        if len(faces) > 0:
            emotion_model_path = 'model.h5'
            EMOTIONS = ["surprised", "disgust", "scared", "happy", "sad", "angry", "neutral"]
            #emotion_classifier = load_model(emotion_model_path, compile=False)
            emotion_classifier = load_model(emotion_model_path)

            for face in faces:
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                face_roi = gray[y:y + h, x:x + w]
                #face_roi = cv2.resize(face_roi, (64, 64))
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi.astype("float") / 255.0
                face_roi = np.expand_dims(face_roi, axis=0)

                preds = emotion_classifier.predict(face_roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]

                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.imshow('Emotion Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 이미지 삭제
            os.remove(filename)
            print(f'{filename} removed!')

        else:
            print("(표정)얼굴을 찾을 수 없습니다.")
#------------------------------------------------------------- 얼굴형 인식
    if key == ord('s'):
        # 현재 시간을 파일 이름으로 사용하여 캡처 저장
        filename = str(time.time()) + '.jpg'
        cv2.imwrite(filename, frame)
        print(f'{filename} saved!')

        # 저장한 파일을 이용하여 얼굴 탐지 및 감정 예측
        image_path = filename
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, axis=-1)

        faces = detector(gray)

        if len(faces) > 0:
            emotion_model_path = 'face-shape-recognizer.h5'
            EMOTIONS = ["square", "round", "oval", "oblong", "heart"]
            #emotion_classifier = load_model(emotion_model_path, compile=False)
            emotion_classifier = load_model(emotion_model_path)

            for face in faces:
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                face_roi = gray[y:y + h, x:x + w]
                #face_roi = cv2.resize(face_roi, (64, 64))
                face_roi = cv2.resize(face_roi, (190, 250))
                face_roi = face_roi.astype("float") / 255.0
                face_roi = np.expand_dims(face_roi, axis=0)

                preds = emotion_classifier.predict(face_roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]

                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.imshow('Emotion Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 이미지 삭제
            os.remove(filename)
            print(f'{filename} removed!')

        else:
            print("(얼굴형)얼굴을 찾을 수 없습니다.")

    # 'q' 키 누르면 종료
    if key == ord('q'):
        break

# 카메라 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
