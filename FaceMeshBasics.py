import cv2
import mediapipe as mp #for efficient processing of multimedia data, enabling real-time applications
import time #for frame rate

print("I am working till here(1)")

#2. access media
cap = cv2.VideoCapture("Videos/10.mp4") 

#pTime is previous time
pTime=0

#mpDraw will help in draw on faces
mpDraw = mp.solutions.drawing_utils

#to create face mesh
mpFaceMesh = mp.solutions.face_mesh

#to find faces that are drawn
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=1)
#Ctrl+click on the function, it'll take you to function details.

#specifying sizes of dots of detected landmarks
drawSpec = mpDraw.DrawingSpec(thickness= 2, circle_radius=1)
#3. run media
#runing video by making it image frames and run side by side as video.
while True:
  success, img = cap.read() #read media
  print("I am working till here(2)")

  #5. Using mediapipe lib to find different points on face
  #faceMesh class only accepts RGB image so we'll first convert this BGR to RBG image
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #sending our RGB img to process in facemesh and store in results
  results = faceMesh.process(imgRGB) 
  #cus of this our fps reduced dramatically
  print("I am working till here(5)")

  #6. displaying results
  #if some landmarks are detected, go ahead and draw them:
  #but we may have multiple faces so first loop thru the faces then draw
  if results.multi_face_landmarks:
    for faceLms in results.multi_face_landmarks:
        mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                              drawSpec,drawSpec)#check fun and play with mpfacemesh. parameters. 
    #In real projects, you need actual locs of points to use them as the project demands. So you can number them to atleast know start/ end of face points:
    #SO you need to loop again for points, right now they are in x,y,z cords(normalized), convert them into pixels. id is index of each cord
        for id, lm in enumerate(faceLms.landmark):
            #print(lm)
            #img_height, img_width, img_channels
            ih,iw, ic= img.shape
            #multiply them with normalized vals to get pixel values, doing for x, y for now
            x,y=int(lm.x*iw), int(lm.y*ih)
            print(id,x,y)

  #4.reading frame rates
  cTime = time.time() #cTime is current time
  fps = 1/(cTime - pTime) 
  pTime = cTime 
  cv2.putText(img, f'FPS: {int(fps)}',(50,70),cv2.FONT_HERSHEY_PLAIN,
              7,(0,0,255),15) #printing fps on img along with text loc,font,scale,color,thickness
  print("I am working till here(3)")

  cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
  cv2.imshow("Image", img)
  cv2.resizeWindow("Image", 500, 700)  # Resize window to 600x400 dimensions
  cv2.waitKey(1)
  print("I am working till here(4)")


#Basics are fine till here, we'll make module out of it now
#Module makes it easier to work as no need for inits and all code written again and again.
