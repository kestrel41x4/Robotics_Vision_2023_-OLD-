from cmath import tan
import cv2
import time
import numpy as np
import math
from pupil_apriltags import Detector
from CoordConstants import VisionConstants

at_detector = Detector(families='tag36h11',
                       nthreads=16,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# Parameters gotten from passing in images to 
# AnalyzeDistortion.py
camera_parameters = np.array([443.6319712,  # fx
                     391.50381628, # fy
                     959.49982957, # cx
                     539.49965467]) # cy

cameraInUse = 0



def focalLengthPx(imageCenter, corner):
    # Get X,Y value of center in np array form 
    center = imageCenter

    
    pixelDistanceY =  (corner[2][1] - corner[1][1])
        
        #print(pixelDistanceY)

    degreesY = (pixelDistanceY/2) * VisionConstants.degreesPerPixel

    radians = (math.pi / 180) * degreesY

    distance = (VisionConstants.tagHeightPx/2) / (math.tan(radians))

    roundedDistance = float("{0:.2f}".format(distance))
    
        
    return -roundedDistance

# Setting up the camera feed
cap = cv2.VideoCapture(cameraInUse)

#TODO: Delete this when using videocapture 
# Setting up camera width and height

while(True):
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Tag size: 0.173m
    tags = at_detector.detect(image, estimate_tag_pose=True, camera_params=camera_parameters, tag_size=0.173)

    
    for tag in tags:
        for p1, p2 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            cv2.line(frame,
                    (int(tag.corners[p1][0]), int(tag.corners[p1][1])),
                    (int(tag.corners[p2][0]), int(tag.corners[p2][1])),
                    (255, 0, 255), 2)
        
      
        h = tag.homography
        
        # Get X,Y value of center in np array form 
        center = tag.center

        # Draws circle dot at the center of the screen
        cv2.circle(frame, (int(center[0]), int(center[1])), radius=8, color=(0, 0, 255), thickness=-1)

        pixelDistanceY1 =  (h[1][1] - h[-1][1])
        
        #print(pixelDistanceY)
        
        pixelDistanceY = -2.71 * (int(pixelDistanceY1))

        degreesY = (pixelDistanceY/2) * VisionConstants.degreesPerPixel

        radians = (math.pi / 180) * degreesY

        distance = (VisionConstants.tagHeightCm/2) / (math.tan(radians))

        roundedDistance = float("{0:.2f}".format(distance))
    
        n = cv2.decomposeHomographyMat(h, camera_parameters)

        
        print(n)
        
        


        

    # Display the resulting frame
    cv2.imshow('Video Feed',frame)
    # cv2.imshow('image',image)

    # The time took to proccess the frame
    endTime = time.monotonic()
    # print(f"{endTime - startTime:.4f}")

    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()