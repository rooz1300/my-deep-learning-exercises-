import cv2
from  mtcnn import MTCNN
import numpy as np
import os 
# TRun off debuging nots 
#os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# cpture video
cap=cv2.VideoCapture(0)
#intance of mtcnn for face detection
detector= MTCNN()
#loading smiledection model 
from tensorflow.keras.models import load_model
net=load_model('smile_net.h5')
#predictions labels
net_label=['not_Similing','Simling']
#function for dection face 
def face_grabber(img):
        
        #coverting from BGR to RGB for face detection 
        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # detecting face with mtcnn
        out=detector.detect_faces(img_rgb)[0]
        # face position 
        x,y,w,h=out['box']
        return img[y:y+h,x:x+w], x,y,w,h, out

while True:
    try:
        ret ,img = cap.read()
        if img is  None: cv2.destroyAllWindows()
        #img=cv2.resize(img, (0,0), fx=1, fy=1)
        # locating fcae 
        detect_faces_for_cnn,x,y,w,h, out=face_grabber(img)
        # preprocsing for smile_net
        detect_faces_for_cnn=cv2.resize(detect_faces_for_cnn, (32,32))/255
        detect_faces_for_cnn=np.array([detect_faces_for_cnn])
        perdiction=net.predict(detect_faces_for_cnn)
        net_label_index=np.argmax(perdiction)
        print(net_label[net_label_index])

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0) )
        cv2.putText(img, net_label[net_label_index], (x-5,y-5),cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
    
        kp=out['keypoints']
        for key,value in kp.items():
            cv2.circle(img,value,2,(21, 232, 183),-1)
        cv2.imshow('image',img)

        if cv2.waitKey(30) == ord('q'):
            break    
   
    except:    
        ret ,img = cap.read()
        img=cv2.resize(img, (0,0), fx=1, fy=1)
        cv2.imshow('image',img)
        if img is  None: break
        
        cv2.imshow('image',img)

        if cv2.waitKey(30) == ord('q'):break

cv2.destroyAllWindows()

cv2.co