import cv2

# classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# video capture setting
capture = cv2.VideoCapture(0) # initialize, # is camera number
capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280) # CAP_PROP_FRAME_WIDTH == 3
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720) # CAP_PROP_FRAME_HEIGHT == 4

face_id = input('\n Registering id를 입력하세요 ==> ')
names = input('\n Registering name을 입력하세요 ==> ').split()
print("\n 얼굴 캡쳐를 시작합니다. 카메라를 봐주세요 ...")

count = 0
#영상 처리 및 출력
while True:
    ret, frame = capture.read() # 카메라 상태 및 프레임
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 흑백
    faces = faceCascade.detectMultiScale(
        gray,# 검출하고자 하는 원본이미지
        scaleFactor = 1.2, # 검색 윈도우 확대 비율
        minNeighbors = 6, # 얼굴 사이 최소 간격
        minSize=(30,30) # 얼굴 최소 크기
    )

    # 얼굴 rectangle 출력
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        count += 1
        # rectangle된 얼굴 jpg형식으로 저장
        cv2.imwrite("dataset/User."+str(face_id)+'.'+str(names)+'.'+str(count)+".jpg", gray[y:y+h, x:x+w])
    cv2.imshow('image',frame)

    # 종료 조건
    if cv2.waitKey(1) > 0 : break # 키 입력이 있을 때 반복문 종료
    elif count >= 100 : break # 100 face sample

print("\n 캡쳐 완료. 프로그램을 종료합니다.")

capture.release() # 메모리 해제
cv2.destroyAllWindows()# 모든 윈도우 창 닫기