import cv2
import numpy as np
from PIL import Image
import os

path = "dataset"  # 경로 (dataset 폴더)
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # 디렉터리 내 파일 리스트로 경로 list 만들기

    faceSamples = []
    ids = []
    for imagePath in imagePaths:  # 각 파일마다
        # 흑백 변환
        PIL_img = Image.open(imagePath).convert('L')  # L : 8 bit pixel
        img_numpy = np.array(PIL_img, 'uint8')

        # user id
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # face sample
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


print('\n dataset 파일의 얼굴 캡쳐를 trainning 중 입니다. 잠시만 기다려주세요 ...')
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))  # trainnig
recognizer.write('trainer/trainer.yml')
print('\n trainning 완료. 프로그램을 종료합니다.'.format(len(np.unique(ids))))
