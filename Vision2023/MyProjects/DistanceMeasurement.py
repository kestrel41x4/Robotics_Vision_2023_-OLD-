# /usr/bin/env python3
import json
from cmath import tan
import cv2
import time
import numpy as np
import math
from pupil_apriltags import Detector
from Constants import VisionConstants

at_detector = Detector(families='tag16h5',
                       nthreads=4,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# Parameters gotten from passing in images to 
# AnalyzeDistortion.py
Cal_file = "LogitechC920.json"
parameters = json.load(open(Cal_file))
#camera_parameters = [parameters["fx"], parameters["fy"], parameters["cx"], parameters["cy"]]
camera_parameters = [606.55440757,
    598.50107753,
    277.1029708,    
    243.19488668]
cameraInUse = 0

# Setting up the camera feed
cap = cv2.VideoCapture(cameraInUse)

#TODO: Delete this when using videocapture 
# Setting up camera width and height
#if cameraInUse == 0:
#cap.set(4, 800)
cap.set(4, 360)

while(True):
    ret, frame = cap.read()
    frame = frame[120:,]
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    val, th = cv2.threshold(image, 80, 255, cv2.THRESH_OTSU)
    # Tag size: 0.152m
    tags = at_detector.detect(th, estimate_tag_pose=False, camera_params=camera_parameters, tag_size=0.152)

    validTags = []

    for tag in tags:
        pixelDistanceY = abs(tag.corners[2][1] - tag.corners[1][1])
        
        
        degreesY = (pixelDistanceY / 2) * VisionConstants.degreesPerPixel
        radians = (math.pi / 180) * degreesY
        tangent = math.tan(radians)
        if abs(tangent) < 1e-6:
            distance = math.inf
        distance = (VisionConstants.tagHeightCm / 2) / (math.tan(radians))
        roundedDistance = float("{0:.2f}".format(distance))

        if tag.tag_id > 8 or tag.tag_id == 0 or pixelDistanceY < 18 or roundedDistance > 500:
            continue
        else:
            validTags.append((tag.tag_id, roundedDistance, 0))
        for p1, p2 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            cv2.line(frame,
                    (int(tag.corners[p1][0]), int(tag.corners[p1][1])),
                    (int(tag.corners[p2][0]), int(tag.corners[p2][1])),
                    (255, 0, 255), 2)
        
        # Get X,Y value of center in np array form 
        center = tag.center

        # Draws circle dot at the center of the screen
        cv2.circle(frame, (int(center[0]), int(center[1])), radius=8, color=(0, 0, 255), thickness=-1)
    # end for tag
    if not validTags:
        continue
    # Display the resulting frame
    cv2.imshow('Video Feed',frame)
    #cv2.imshow('Video Feed', image)
    #cv2.imshow('Video Feed', image_gray)
    #qcv2.imshow('Video Feed', th)

    print(tag.corners)
    # cv2.imshow('image',image)

    # The time took to proccess the frame
    endTime = time.monotonic()
    # print(f"{endTime - startTime:.4f}")

    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
