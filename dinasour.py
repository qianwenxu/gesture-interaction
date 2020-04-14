import numpy as np
import cv2
from threading import Event
import re
import math
import time
import random
import os
import win32api
import win32con
import win32gui

def _remove_background(img):#去除背景
    fgbg = cv2.createBackgroundSubtractorMOG2() # 利用BackgroundSubtractorMOG2算法消除背景
    fgmask = fgbg.apply(img)
    #cv2.imshow("image2", fgmask)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(img, img, mask=fgmask)
    return res

def _bodyskin_detetc(frame):#得到去除背景的图片skin
    # 肤色检测: YCrCb之Cr分量 + OTSU二值化
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) # 分解为YUV图像,得到CR分量
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0) # 高斯滤波
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # OTSU图像二值化
    cv2.imshow("image1", skin)
    return skin

def _get_contours(array):#得到图片所有的坐标
    contours, hierarchy = cv2.findContours(array, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def _get_eucledian_distance(beg, end):#计算两点之间的坐标
    i=str(beg).split(',')
    j=i[0].split('(')
    x1=int(j[1])
    k=i[1].split(')')
    y1=int(k[0])
    i=str(end).split(',')
    j=i[0].split('(')
    x2=int(j[1])
    k=i[1].split(')')
    y2=int(k[0])
    d=math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    return d
    
def _get_defects_count(array, contour, defects, verbose = False):
    ndefects = 0
    for i in range(defects.shape[0]):
        s,e,f,_= defects[i,0]
        beg= tuple(contour[s][0])
        end= tuple(contour[e][0])
        far= tuple(contour[f][0])
        a= _get_eucledian_distance(beg, end)
        b= _get_eucledian_distance(beg, far)
        c= _get_eucledian_distance(end, far)
        angle= math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) # * 57
        if angle <= math.pi/2 :#90:
            ndefects = ndefects + 1
            if verbose:
                cv2.circle(array, far, 3, _COLOR_RED, -1)
                cv2.imshow("image2", array)
        if verbose:
            cv2.line(array, beg, end, _COLOR_RED, 1)
            cv2.imshow("image2", array)
    return array, ndefects

def grdetect(array, verbose = False):
    copy = array.copy()
    array = _remove_background(array) # 移除背景, add by wnavy
    thresh = _bodyskin_detetc(array)
    contours = _get_contours(thresh.copy()) # 计算图像的轮廓
    largecont= max(contours, key = lambda contour: cv2.contourArea(contour))
    hull= cv2.convexHull(largecont, returnPoints = False) # 计算轮廓的凸点
    defects= cv2.convexityDefects(largecont, hull) # 计算轮廓的凹点
    if defects is not None:
        # 利用凹陷点坐标, 根据余弦定理计算图像中锐角个数
        copy, ndefects = _get_defects_count(copy, largecont, defects, verbose = verbose)
        # 根据锐角个数判断手势, 会有一定的误差
        if   ndefects == 0:
            print("1根或0根")
        elif ndefects == 1:
            print("2根")
        elif ndefects == 2:
            print("3根")
        elif ndefects == 3:
            print("4根")
        elif ndefects == 4:
            print("5根")
            win32api.keybd_event(0x20,0,0,0)
            win32api.keybd_event(0x20,0,win32con.KEYEVENTF_KEYUP,0)
            print('跳一次')

def judge():
    imname =  "wif.jpg"
    img = cv2.imread(imname, cv2.IMREAD_COLOR)
    grdetect(img, verbose = False)
    
def main():
    capture = cv2.VideoCapture(0)#打开笔记本的内置摄像头
    cv2.namedWindow("camera",1)
    start_time = time.time()
    while(1):
        ha,img =capture.read()
        #按帧读取视频，ha,img为获取该方法的两个返回值，ha为布尔值，如果读取帧是正确的则返回true
        #如果文件读取到结尾，它的返回值就为false,img就是每一帧的图像，是个三维矩阵
        end_time = time.time()#返回当前时间的时间戳
        cv2.rectangle(img,(326,0),(640,310),(170,170,0))
        #cv2.putText(img,str(int((10-(end_time- start_time)))), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        #对应参数为图片、添加的文字，左上角图标，字体，字体大小，颜色，即上面一行功能为在图片中显示倒计时
        cv2.imshow("camera",img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            print('结束')
            break
        ha,img = capture.read()
        cv2.imshow("camera",img)
        img = img[0:310,326:640]
        cv2.imwrite("wif.jpg",img)
        judge()
    capture.release()
    cv2.destroyAllWindows()

main()