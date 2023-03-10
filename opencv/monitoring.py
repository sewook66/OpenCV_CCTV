import cv2
import numpy as np
import winsound as ws


def beepsound(): # beep sound
    freq = 2000
    dur = 1000
    ws.Beep(freq, dur)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX
names = input('\n Registered name을 입력하세요 ==> ').split()
print('\n')
id = 0

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

minW = 0.1 * cam.get(cv2.CAP_PROP_FRAME_WIDTH)
minH = 0.1 * cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 55:
            id = names[id]
            print(" detected")
            beepsound()
            count += 1
            cv2.imwrite("detected/User." + str(id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        else:
            id = "unknown"

        confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    if cv2.waitKey(1) > 0: break
    elif count >= 10:
        print("\n 감지가 되어 10번의 캡쳐가 실행되었습니다")
        break

print("\n 프로그램을 종료합니다.")
cam.release()
cv2.destroyAllWindows()