
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


# In[2]:


def siftTrack(trainImg,trainKP,trainDesc,QueryImgBGR,h,w):
    MIN_MATCH_COUNT=10

    detector=cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDITREE=0
    flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
    searchParam = dict(checks=50)
    flann=cv2.FlannBasedMatcher(flannParam,searchParam)

    #trainImg=cv2.imread("demo0.png",0)
    #trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

    #cam=cv2.VideoCapture(0)
    #while True:
    #ret, QueryImgBGR=cam.read()
    
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)

    goodMatch=[]
    queryBorder = None
    for m,n in matches:
        if(m.distance < 0.7*n.distance):
            goodMatch.append(m)
    if(len(goodMatch) > MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        print(H)
        
        #h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
        return (1,queryBorder)
        #print(queryBorder)
    else:
        #print("Not Enough match found")
        return (-1,queryBorder)



# In[ ]:

'''
cam = cv2.VideoCapture(0)
img1 = cv2.imread("demo0.png")
img3 = cv2.imread("demo0.png")
h,w,d = img1.shape
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
while True:
    ret, img2=cam.read()
    checker,dst = siftTrack(img1,kp1,des1,img2,h,w)
    if checker != -1:
        img3 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        print("YES")
    else:
        print("NO")
    keypressed = cv2.waitKey(5)
    if keypressed == 27:
        break
    cv2.imshow('mask',img3)
    
cam.release()
cv2.destroyAllWindows()
'''
