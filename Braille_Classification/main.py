from matplotlib import pyplot as plt
from keras.models import Model,load_model
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import glob


# Bitplane Slicing 함수
def bitSlice(img, path):
 # 크기가 같은 임의의 밝기 최대의 이미지 생성
 plane = np.full((img.shape[0], img.shape[1]), 255, np.uint8)
 # Bitwise And 실행
 res = cv2.bitwise_and(plane, img)
 # 너무 구분이 안되서 255밝기 곱함
 x = res * 255
 # Erode 위한 Kernel (점이 검은색이라 확장시키려면 Erode해야 함)
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
 # Erode해서 이미지 정리
 morphimage = cv2.erode(x, kernel, iterations=3)
 ret, thr1  = cv2.threshold(morphimage, 10, 255, cv2.THRESH_BINARY)
 cv2.imwrite(path, thr1)

# 선(버튼)을 지우는 함수
def deleteLine(img):
 # canny로 edge 검색
 edges = cv2.Canny(img,10,230)
 # Houghlinep으로 직선 정의
 lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=20,maxLineGap=10)

 if(lines.any() != None):
 #직선 긋기
  for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(230,230,230),5)

 return img

model = load_model('BrailleNet.h5')
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
path = glob.glob('./before_preprocess/*.png')
#***************위의 경로의 png 파일들이 test data입니다.**********#
#1. test data를 살펴봅니다.
#2.리눅스 터미널에서 main.py를 실행하여 테스트를 진행합니다. 
#3. main.py를 실행 시킨 후 전처리된 이미지에 대해서 살펴봅니다. 
##################################################################

path.sort()
score=0
for p in path:
      
  img = cv2.imread(p)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  name = p.split('/')[-1]
  answer = name
  answer= answer.split('.')[0]
  #print(name)
  name = './after_preprocess/' + name+'_temp.png'

  deleteLine(img)
  bitSlice(img, name)



  im = cv2.imread(name)

  params = cv2.SimpleBlobDetector_Params()

  #filter by color????    black dots

  # Filter by Area.
  params.filterByArea = True
  params.minArea = 12

  # Filter by Circularity
  params.filterByCircularity = True
  params.minCircularity = 0.6

  # Filter by Convexity
  params.filterByConvexity = True
  params.minConvexity = 0.6

  # Filter by Inertia
  params.filterByInertia = True
  params.minInertiaRatio = 0.4

  # Create a detector with the parameters
  ver = (cv2.__version__).split('.')

  if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
  else :
    detector = cv2.SimpleBlobDetector_create(params)

  keypoints = detector.detect(im)

  x = []
  y = []

  for keyPoint in keypoints:
      x.append(keyPoint.pt[0])
      y.append(keyPoint.pt[1])
      #s = keyPoint.size

  coordinates = np.array(list(zip(x,y)))
  #print(coordinates)

  # Draw detected blobs as red circles.

  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

  im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


  img = cv2.imread(name)

  x.sort()
  y.sort()

  xcord = int(x[0])
  ycord = int(y[0])
  w = int(x[-1] - x[0])
  h = int(y[-1] - y[0])

  cropped_img = img[ycord-15: ycord + h+15, xcord-15: xcord + w+15]


  braille_dict ={0:'1' , 1:'10' , 2:'11', 3:'12', 4:'13',5:'14',6:'15',7:'16',8:'17',9:'18',10:'19',11:'2',12:'3',13:'4',14:'5',15:'6',16:'7',17:'8', 18:'9', 19:'closing', 20:'opening'}


  img = cv2.resize(cropped_img, (28,28))
  #img = Image.open('./images2/1/1_0_20.jpg')
  img = np.array(img)
  img = (np.expand_dims(img,0))
  predictions = probability_model.predict(img)
  
  if answer == braille_dict[predictions[0].argmax(0)]:
        score+=1
  print('입력된 점자의 값 : ' +answer , end='   ')
  print('예측된 점자는 : '+ braille_dict[predictions[0].argmax(0)])
print('21개의 데이터 중 정답 개수 ' + str(score))