#pip install mediapipe==0.8.3.1 --user

#FACE_CONNECTIONS ---- > FACEMESH_TESSELATION

#%%
import cv2
import mediapipe as mp
import numpy as np
import time
from ImageStack import stackImages

#%%
pT=0
mpPose=mp.solutions.holistic
mpDraw=mp.solutions.drawing_utils

face_drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
face_connect_spec=mpDraw.DrawingSpec(thickness=2, circle_radius=1,color=(255,254,255))

left_hand_drawing_spec = mpDraw.DrawingSpec(thickness=2, circle_radius=2)
left_hand_connect_spec=mpDraw.DrawingSpec(thickness=2, circle_radius=2,color=(155,255,155))

right_hand_drawing_spec = mpDraw.DrawingSpec(thickness=2, circle_radius=3)
right_hand_connect_spec=mpDraw.DrawingSpec(thickness=2, circle_radius=2,color=(155,255,155))

pose_drawing_spec = mpDraw.DrawingSpec(thickness=2, circle_radius=3)
pose_connect_spec=mpDraw.DrawingSpec(thickness=2, circle_radius=2,color=(155,255,155))

#%%
def Pose_prediction(image,black_f):
    points=[]
    imgRGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    if (results.pose_landmarks):
        #mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        mpDraw.draw_landmarks(image, results.face_landmarks, mpPose.FACEMESH_TESSELATION,landmark_drawing_spec=face_drawing_spec,connection_drawing_spec=face_connect_spec)
        mpDraw.draw_landmarks(image, results.left_hand_landmarks, mpPose.HAND_CONNECTIONS,landmark_drawing_spec=left_hand_drawing_spec,connection_drawing_spec=left_hand_connect_spec)
        mpDraw.draw_landmarks(image, results.right_hand_landmarks, mpPose.HAND_CONNECTIONS,landmark_drawing_spec=right_hand_drawing_spec,connection_drawing_spec=right_hand_connect_spec)
        mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS,landmark_drawing_spec=pose_drawing_spec,connection_drawing_spec=pose_connect_spec)
        
        mpDraw.draw_landmarks(black_f, results.face_landmarks, mpPose.FACEMESH_TESSELATION,landmark_drawing_spec=face_drawing_spec,connection_drawing_spec=face_connect_spec)
        mpDraw.draw_landmarks(black_f, results.left_hand_landmarks, mpPose.HAND_CONNECTIONS,landmark_drawing_spec=left_hand_drawing_spec,connection_drawing_spec=left_hand_connect_spec)
        mpDraw.draw_landmarks(black_f, results.right_hand_landmarks, mpPose.HAND_CONNECTIONS,landmark_drawing_spec=right_hand_drawing_spec,connection_drawing_spec=right_hand_connect_spec)
        mpDraw.draw_landmarks(black_f, results.pose_landmarks, mpPose.POSE_CONNECTIONS,landmark_drawing_spec=pose_drawing_spec,connection_drawing_spec=pose_connect_spec)
        
        for idx,lm in enumerate(results.pose_landmarks.landmark):
                     h,w,c=img.shape
                     cx,cy=int(lm.x*w),int(lm.y*h)
                     points.append([idx,cx,cy])
    return image,black_f,points


#%%
###For Video
#Add path of a video if want
#cap=cv2.VideoCapture("Solo - Clean Bandit ft. Demi Lovato  Jane Kim Choreography.mp4")

###For Webcam
# 0--> Default Webcam    
#%%
cap=cv2.VideoCapture(0)

pose=mpPose.Holistic()
while True:               
  ret,img=cap.read()
  #frame = cv2.resize(frame, (640,480))
  img=cv2.resize(img,(0,0),fx=0.7,fy=0.7)
  black_f = np.zeros(img.shape, dtype=np.uint8)
  if ret==0:
      break
  img=cv2.flip(img,1)
  img,black_frame,points=Pose_prediction(img,black_f)
  #black_frame=Pose_prediction(frame0)
  
  cT=time.time()
  fps=1/(cT-pT)
  pT=cT
  if len(points):
      #NECK
      dist=int(abs(0.8*(points[2][2]-points[9][2])))
      start=(int((points[10][1]+points[9][1])/2),(int((points[10][2]+points[9][2])/2)+dist))
      stop=(int((points[11][1]+points[12][1])/2),(int((points[11][2]+points[12][2])/2)))
      cv2.line(img,start,stop,(155,255,155),2)
      cv2.line(black_frame,start,stop,(155,255,155),2)
      
      #BELLY
      Belly_col=(0,255,255)
      belly_1=points[11][1:]
      belly_2=points[12][1:]
      belly_3=points[23][1:]
      belly_4=points[24][1:]
      belly_cnt = np.array([belly_1, belly_2, belly_3,belly_4])
      #cv2.drawContours(img, [belly_cnt], 0, Belly_col, -1)
      cv2.drawContours(black_frame, [belly_cnt], 0, Belly_col, -1)
      
      #LEG1
      Leg_col=(0,0,255)
      Leg_11=points[27][1:]
      Leg_12=points[29][1:]
      Leg_13=points[31][1:]
      Leg_1_cnt = np.array([Leg_11, Leg_12, Leg_13])
      #cv2.drawContours(img, [Leg_1_cnt], 0, (0,255,0), -1)
      cv2.drawContours(black_frame, [Leg_1_cnt], 0,Leg_col, -1)  
      
       #LEG2
      Leg_21=points[28][1:]
      Leg_22=points[30][1:]
      Leg_23=points[32][1:]
      Leg_2_cnt = np.array([Leg_21, Leg_22, Leg_23])
      #cv2.drawContours(img, [Leg_2_cnt], 0, (0,255,0), -1)
      cv2.drawContours(black_frame, [Leg_2_cnt], 0,Leg_col, -1)  
  
  
  cv2.putText(img,str(int(fps))+" fps",(18,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
  imageArray = ([img,black_frame])
  stackedImage = stackImages(imageArray, 1)
  cv2.imshow('LIVE', stackedImage)

  k=cv2.waitKey(1)
  if(k==27):
      break
  
cv2.destroyAllWindows() 
cap.release() 
  
  
#   if results.pose_landmarks:
#       mp_drawing.draw_landmarks(frame, results.pose_landmarks, conn, drawing_spec2, drawing_spec1)
#       mp_drawing.draw_landmarks(frame0, results.pose_landmarks, conn, drawing_spec2, drawing_spec1)
  
#   cv2.circle(frame0,(10,10),10,2) 
  
#   cv2.imshow('MediaPipe Pose', frame)
#   cv2.imshow('MediaPipe Pose black', frame0)
#   if cv2.waitKey(1) & 0xFF == 27:
#     break
# cv2.destroyAllWindows()
# cap.release()