# OpenCV_CCTV

### 요약
Pycharm Community 에서 Python 과 OpenCV, Haar Cascade Classifier library를 활용하여 만든 CCTV. <br>
<table>
  <td>
    <img width="250" height="100" alt="Python_logo" src="https://user-images.githubusercontent.com/51785417/224247593-0746d6db-9c44-49a5-9ac4-f861b0d7b3f3.png">
  </td>
  <td>
    <img width="250" height="250" alt="OpenCV_logo" src="https://user-images.githubusercontent.com/51785417/222953853-27ed8abd-f1a1-4241-a3d4-46a66cb2ed9d.png">
  </td>
</table>
연결된 camera로 인식된 대상의 이미지에 id와 name를 라벨링후 학습하여 monitoring 단계에서 detecting을 해서 포착 인물 캡쳐 및 beep가 울린다. <br>
총 3개의 파일로 datasetting, trainning, monitoring 을 따로 수행한다. <br>

- datasetting <br>
연결된 camera를 통해 사람의 얼굴을 인식을 위한 OpenCV의 미리 학습시켜져 있는 분류기(Cascade Classifier)를 사용한다 그중에서도 사람의 전체적인 얼굴을 인식하고 검출된 이미지 픽셀을 BGR로 받아온 이미지를 흑백으로 처리 해주는 haarcascade_fontalface_default로 scalefactor, minneighbors, minsize를 조정해준다. <br>
검출된 이미지를 캡쳐해 trainning 시키기위해 초기에 id와 name을 부여해 프레임마다 한장씩 총 100장의 jpg파일 표본으로 만들어 User.(id).[‘name’].(count)형태로 dataset파일에 저장. <br>

- trainning <br>
dataset파일의 경로를 받아 저장된 jpg파일 표본을 갑져와 배열에 담는다. <br>
경로를 설정하고 LBPHFaceRecognizer를 생성한뒤 학습을 시켜서 trainer.yml 파일로 trainer 폴더에 생성 <br>

- monitoring <br>
trainning.py 에서 학습시킨 recognizer을 가져와서 user id에 해당하는 name을 입력해 지정해준다.
실행시 datasetting 때와 같이 카메라가 setting되며 화면에서 recognizer로 얼굴을 예측하고, <br>
학습되어 있는 얼굴의 이미지와 매칭하여 확률값이 55% 이상이 매칭된다면 초기에 작성한 name이 화면에 나타나면서 <br>
detected 파일에 캡쳐가 되며 beep가 울린다. <br>
trainning 되지않아 매칭 확률값이 55% 미만인 대상이 비춘다면 unknown으로 인식을 한다.
