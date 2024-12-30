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

        self.drawSpec = self.mpDraw.DrawingSpec(thickness= 2, circle_radius=1)
    
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
                                4,(0,0,255),5)
                    #print(id,x,y)
                    face.append([x,y])
                #storing these in the list
                faces.append(face) 
        return img, faces

#For Image Detection
def main_image():
    img = cv2.imread("Videos/14.jpg")  
    detector = FaceMeshDetector()
    img, faces = detector.findFaceMesh(img)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  
    cv2.imshow("Image", img)
    cv2.resizeWindow("Image", 500, 700)
    cv2.waitKey(0) 

#1. What to do if you are running this module by itself:
if __name__=="__main__":
    # Uncomment the desired function call
    #main_video() 
    main_image()