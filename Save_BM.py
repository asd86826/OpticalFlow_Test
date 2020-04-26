import cv2
import numpy as np

camera = cv2.VideoCapture('GF.mp4')

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('OF_BM.mp4', fourcc, 30.0, (540, 960))

# 构建椭圆结果
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
kernel = np.ones((5, 5), np.uint8)
background = None

while True:
    # Read video
    grabbed, frame_lwpCV = camera.read()

    # 对帧进行预处理，>>转灰度图>>高斯滤波（降噪：摄像头震动、光照变化）。
    gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

    # 将第一帧设置为整个输入的背景
    if background is None:
        background = gray_lwpCV
        continue

    # 对比背景之后的帧与背景之间的差异，并得到一个差分图（different map）。
    # 阈值（二值化处理）>>膨胀（dilate）得到图像区域块
    diff = cv2.absdiff(background, gray_lwpCV)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, es, iterations=2)

    # 显示矩形框：计算一幅图像中目标的轮廓
    contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 1500:      # 对于矩形区域，只显示大于给定阈值的轮廓（去除微小的变化等噪点）
            continue
        (x, y, w, h) = cv2.boundingRect(c) # 该函数计算矩形的边界框
        cv2.rectangle(frame_lwpCV, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('contours', frame_lwpCV)
    cv2.imshow('dis', diff)
    out.write(diff)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalBMfb.png',frame_lwpCV)
        cv2.imwrite('opticaldi.png',diff)
# 释放资源并关闭窗口
camera.release()
cv2.destroyAllWindows()
out.release()
