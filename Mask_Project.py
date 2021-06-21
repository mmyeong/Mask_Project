from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model #모델 로드


from imutils.video import FPS
import imutils
import time
import argparse
import numpy as np
import cv2
import os

#얼굴 인식 및 Mask 객체
def mask_detector(frame,model):
    #프레임 크기
    (h, w) = frame.shape[:2]

    src = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    #얼굴 인식
    network.setInput(src) #setInput 이미지를 네트워크의 입력으로 설정
    detections = network.forward() #얼굴 인식

    faces = [] #얼굴
    locations = [] #좌표
    predicts = [] #확률

    #얼굴 인식을 위한 반복
    for i in range(0, detections.shape[2]):
        #얼굴 인식 확률 추출
        confidence = detections[0, 0, i,2]

        #얼굴 인식 확률이 최소 확률 보다 큰 경우
        if confidence > minimum_confidence:
            #bounding box 위치 계산
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")

            # bounding box 가 전체 좌표 내에 있는지 확인
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            #인식된 얼굴 추출 후 전처리
            face = frame[startY:endY, startX:endY] #인식된 얼굴 추출
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) #RGB 변환
            face = cv2.resize(face, (224,224))#얼굴 크기 조정
            face = img_to_array(face)
            face = preprocess_input(face)# 이미지를 적절하게 맞추기 함수

            #전처리된 얼굴 이미지 및 좌표 목록 추가
            faces.append(face)
            locations.append((startX, startY,endX,endY))

    #face 1개 이상 감지된 경우
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        predicts = model.predict(faces, batch_size=32)

    #얼굴 위치 및 확률 반환
    return (locations, predicts)


#실행 할 때 인자값 추가
ap = argparse.ArgumentParser() #인자값 받을 인스턴스 생성
#입력 받을 인자값
ap.add_argument("-i","--input",type=str, help="input 비디오 경로")
ap.add_argument("-o","--output",type=str, help="output 비디오 경로")
#입력받은 인자값을 args에 저장

# #얼굴 인식모델 미완성
# face_detector = "./face_detector/"
# prototxt = face_detector + "deploy.prototxt"
# weights = face_detector + "res10_300x300_ssd_iter_140000.caffemodel"
# network = cv2.dnn.readNet(prototxt, weights) # cv2.dnn.readNet() : 네트워크를 메모리에 로드
#

#Mask Detector 모델
mask_detector_model = "mask_model.h5"
model = load_model(mask_detector_model)
print('모델 불러오기 성공')

#카메라
capture = cv2.VideoCapture(0)

#캡쳐 프레임 사이즈 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH,320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

# 인식할 최소 확률
minimum_confidence = 0.5

writer = None

fps = FPS().start()

#비디오 스트림 프레임 반복
mask_cnt = 1 #마스크 착용 상태 확인
while True:
    ret,frame = capture.read()
    if ret == True:
        print('성공')
    # 이미지 뒤집기
    frame_fliped = cv2.flip(frame, 1)
    # 이미지 출력
    cv2.imshow('VideoFrame', frame_fliped)
    # 1초마다 검사하며, 아무키나 누르면 종료
    if cv2.waitKey(200) > 0:
        break

    (locations,predicts) = mask_detector(frame,model)

    #인식된 얼굴 수 만큼 반복
    for (box, predict) in zip(locations,predicts):
        (startX,startY,endX,endY) = box
        #With_Mask 0
        #No_Mask 1
        # without_mask : 마스크 미착용 확률
        (mask, without_mask) = predict  # 확률
        # bounding box 레이블 설정
        label = "Mask" if mask > without_mask else "No Mask"

        # bounding box 색상 설정
        if label == "Mask" and max(mask, without_mask) * 100 >= 70:
            color = (0, 255, 0)  # 초록
        elif label == "No Mask" and max(mask, without_mask) * 100 >= 70:
            color = (0, 0, 255)  # 빨강
        else:
            color = (0, 255, 255)  # 노랑

        # 확률 설정
        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        # bounding box 출력
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # fps 정보 업데이트
        fps.update()

# fps 정지 및 정보 출력
fps.stop()
print("[재생 시간 : {:.2f}초]".format(fps.elapsed()))
print("[FPS : {:.2f}]".format(fps.fps()))

# 종료
capture.release()
cv2.destroyAllWindows()


