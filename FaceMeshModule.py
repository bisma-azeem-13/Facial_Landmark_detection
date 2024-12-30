import cv2
import mediapipe as mp 
import time

class FaceMeshDetector():
    #we'll give parameters that we had in FaceMesh()
    def __init__(self, staticMode=False, maxFaces=5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
    
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, 
                                                min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.drawSpec = self.mpDraw.DrawingSpec(thickness= 1,color=(0,255,0), circle_radius=1)
      
        #inside class, findmesh function:
    def findFaceMesh(self, img, draw=True): #draw is flag, draw or not draw

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results =self.faceMesh.process(self.imgRGB) 

        faces=[] #as many faces
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw: #as draw is flag, optional
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                              self.drawSpec,self.drawSpec)
                face=[]
                for id, lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih,iw, ic= img.shape
                    x,y=int(lm.x*iw), int(lm.y*ih)
                    #printing id number on media
                    cv2.putText(img, str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,
                               0.1,(0,0,255),1)
                    #print(id,x,y)
                    face.append([x,y])
            #storing these in the list
                faces.append(face)  
        return img, faces


#Wherever its giving error just put self there.
def main():
  cap = cv2.VideoCapture(0) #index 0 gives access to webcam for real-time detection
  # cap = cv2.VideoCapture("Videos/10.mp4") for path-wise detection


  pTime=0
  #calling fun to do its work
  detector = FaceMeshDetector()
  while True:
    success, img = cap.read() #read media
    img, faces = detector.findFaceMesh(img)
    #print("Number of faces detected: ")
    #if len(faces)!=0:
    #    print(len(faces))
    cTime = time.time() #cTime is current time
    fps = 1/(cTime - pTime) 
    pTime = cTime 
    cv2.putText(img, f'FPS: {int(fps)}',(50,70),cv2.FONT_HERSHEY_PLAIN,
    2,(0,0,255),2) #printing fps on img along with text loc,font,scale,color,thickness

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    cv2.imshow("Image", img)
    #cv2.resizeWindow("Image")  # Resize window to 600x400 dimensions
    cv2.waitKey(1)
    

#1. What to do if you are running this module by itself:
if __name__=="__main__":
    main()