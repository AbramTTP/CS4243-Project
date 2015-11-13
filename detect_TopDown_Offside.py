#Authors: Muhammad Fazli Bin Sapuan, Tan Tack Poh

import cv2
import sys
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt

filename = 'output.avi'
cap = cv2.VideoCapture(filename)

width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CV_CAP_PROP_FPS))
count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT ))

print "width:\t%d " % width
print "height:\t%d " % height
print "fps:\t%d " % fps
print "count:\t%d " % count
sys.stdout.flush()

bg = cv2.imread('bg.png')
topDown = cv2.imread('field.png')

height, width, channels = bg.shape
mask = np.zeros((height,width,channels), np.uint8)
# Four corners of the field
ptsInt = np.array([[1948, 122], [3099, 113], [432, 538], [4829, 509]],np.int32)
ptsInt = ptsInt.reshape((-1,1,2))
cv2.fillPoly(mask, [ptsInt], (255, 255, 255))

pts = np.array([[1948, 122], [4829, 509], [432, 538], [3099, 113]],np.float32)
#pts = pts.reshape((-1,1,2))

ptsTD = np.array([[25, 10], [570, 365], [25, 365], [570, 10]],np.float32)
#ptsTD = pts.reshape((-1,1,2))

fieldToTDPerspMat = cv2.getPerspectiveTransform(pts, ptsTD)
TDToFieldPerspMat = cv2.getPerspectiveTransform(ptsTD, pts)

for fr in range(1,count):

    lastDefenderRed = 10000
    secondLastDefenderRed = 10000
    
    lastDefenderBlue = 0
    secondLastDefenderBlue = 0

    topY = np.float32(12)
    bottomY = np.float32(365)

    newTopDown = topDown.copy()
    _,img = cap.read()
    absdiff = cv2.absdiff(img, bg)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32,32))
    masked_img = cv2.bitwise_and(mask, absdiff)
    hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    h = cv2.equalizeHist(h)
    s = cv2.equalizeHist(s)
    v = clahe.apply(v)
    _, v = cv2.threshold(v, 70, 255, cv2.THRESH_BINARY)
    v = cv2.GaussianBlur(v, (7,33),0)
    _, v = cv2.threshold(v, 70, 255, cv2.THRESH_BINARY)
    vvv = cv2.merge((v,v,v))
##    masked_img = cv2.bitwise_and(img, vvv)
##    r = masked_img[:, :, 0]
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    contours, hierarchy = cv2.findContours(v,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] == 0: continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if cy > 450: continue
        area = cv2.contourArea(contour)
        if area < 30: continue
        contour_mask = np.zeros(v.shape,np.uint8)
        cv2.drawContours(contour_mask,[contour],0,(255,255,255),-1)
        mean = cv2.mean(hsv, mask = contour_mask)
        if (0, 50, 0) < mean < (35, 255, 85):
            # found red player
            color = (0, 0, 255)
        elif (35, 0, 0) < mean < (80, 50, 85):
            # found blue player
            color = (255, 0, 0)
        else:
            color = mean
        cv2.circle(img, (cx, cy), 20, color, 2)
        cx = np.float32(cx + 15)
        cy = np.float32(cy + 15)
        srcPts = np.array([[[cx, cy]]], dtype = "float32")
        destPt = cv2.perspectiveTransform(srcPts, fieldToTDPerspMat)       
        topDownPt = (destPt[0][0][0], destPt[0][0][1])
        cv2.circle(newTopDown, topDownPt, 5, color, -1)

        if(color == (0, 0, 255)):
            if(destPt[0][0][0] < lastDefenderRed):
                lastDefenderRed = destPt[0][0][0]
                secondLastDefenderRed = lastDefenderRed
            elif(destPt[0][0][0] < secondLastDefenderRed):
                secondLastDefenderRed = destPt[0][0][0]
        elif(color == (255, 0, 0)):
            if(destPt[0][0][0] > lastDefenderBlue):
                lastDefenderBlue = destPt[0][0][0]
                secondLastDefenderBlue = lastDefenderBlue
            elif(destPt[0][0][0] > secondLastDefenderBlue):
                secondLastDefenderBlue = destPt[0][0][0]

        srcPtsRedTopPt = np.array([[[secondLastDefenderRed, topY]]], dtype = "float32")
        srcPtsRedBottomPt = np.array([[[secondLastDefenderRed, bottomY]]], dtype = "float32")
        srcPtsBlueTopPt = np.array([[[secondLastDefenderBlue, topY]]], dtype = "float32")
        srcPtsBlueBottomPt = np.array([[[secondLastDefenderBlue, bottomY]]], dtype = "float32")

        destPtsRedTopPt = cv2.perspectiveTransform(srcPtsRedTopPt, TDToFieldPerspMat)
        destPtsRedBottomPt = cv2.perspectiveTransform(srcPtsRedBottomPt, TDToFieldPerspMat)
        destPtsBlueTopPt = cv2.perspectiveTransform(srcPtsRedTopPt, TDToFieldPerspMat)
        destPtsBlueBottomPt = cv2.perspectiveTransform(srcPtsRedBottomPt, TDToFieldPerspMat)

        ptsRedTopPt = (destPtsRedTopPt[0][0][0], destPtsRedTopPt[0][0][1])
        ptsRedBottomPt = (destPtsRedBottomPt[0][0][0], destPtsRedBottomPt[0][0][1])
        ptsBlueTopPt = (destPtsBlueTopPt[0][0][0], destPtsBlueTopPt[0][0][1])
        ptsBlueBottomPt = (destPtsBlueBottomPt[0][0][0], destPtsBlueBottomPt[0][0][1])

        cv2.line(img, ptsRedTopPt, ptsRedBottomPt, (0, 0, 255), 3, 8)
        cv2.line(img, ptsBlueTopPt, ptsBlueBottomPt, (255, 0, 0), 3, 8)

    cv2.imshow("output", img)
    cv2.imshow("output2", newTopDown)
    cv2.waitKey(10)
