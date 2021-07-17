import cv2
import tensorflow.keras
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import playsound
import random
import os
def speak(text):
    tts = gTTS(text=text, lang='ko')
    # 파일 이름 설정
    filename = 'mmyeong.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

#이미지 전처리
def preprocessing(frame):
    #사이즈 조정
    size = (224,224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    #이미지 정규화
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

    #이미지 차원 재조정 - 예측을 위해 reshape
    frame_reshaped = frame_normalized.reshape((1,224,224,3))

    return frame_reshaped

#학습된 모델 불러오기
model_filename = 'mask_model.h5'
model = tensorflow.keras.models.load_model(model_filename)

#카메라
capture = cv2.VideoCapture(0)

#캡쳐 프레임 사이즈 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH,320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

sleep_cnt = 1 #20초간 '졸음'상태 확인
while True:
    ret, frame = capture.read()
    if ret == True:
        print('성공')

    #이미지 뒤집기
    frame_fliped = cv2.flip(frame,1)

    #이미지 출력
    cv2.imshow('VideoFrame',frame_fliped)

    #1초마다 검사하며, 아무키나 누르면 종료
    if cv2.waitKey(200)>0:
        break

    #데이터 전처리
    preprocessed = preprocessing(frame_fliped)

    #예측
    prediction = model.predict(preprocessed)

    if prediction[0,0] > prediction[0,1]:
        print('mask 착용')
        sleep_cnt += 1
    else:
        speak('마스크를 착용 해주세요')
        print('mask 착용 안함')
        sleep_cnt=1

# 카메라 객체 반환
capture.release()
# 화면에 나타난 윈도우들을 종료
cv2.destroyAllWindows()
