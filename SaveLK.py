import numpy as np
import cv2
import time


cap = cv2.VideoCapture('X+.mp4')

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('1OF_X-.mp4', fourcc, 30.0, (540, 960))


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,        # How many point you wnat to check 
                       qualityLevel = 0.3,      # The point quality
                       minDistance = 70,        # Point repeating range
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))


# Take first frame and find corners in it
ret, old_frame = cap.read()                     # Catch the  
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)


while(1):
    Vx , Vy = 0, 0 

    Start_time = time.time()

    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    try:
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    except Exception as e:
        print (e)
        # If the point out of range ,find a point again
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

        Vx += c - a
        Vy += d - b
        number = i


    Vx = np.float32(270+Vx/(number+1))
    Vy = np.float32(480+Vy/(number+1))
    mask = cv2.line(mask, (270,480),(Vx,Vy), color[0].tolist(), 2)

    img = cv2.add(frame,mask)

    out.write(img)

    cv2.imshow('frame',img)

    print ("\nXVelocity : %f \t\tY : %f" %(Vx , Vy))

    # Press the Esc to close
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',img)

cv2.destroyAllWindows()
cap.release()
out.release()
