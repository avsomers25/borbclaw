import pyautogui
import time

import cv2
import numpy as np

from mss import mss

sct = mss() 

bounding_box = {'top': 275, 'left': 275, 'width': 1325, 'height': 475}    

template = cv2.imread('claw.png')

filename = sct.shot()
print(filename)  

def find_claw(sct_img_np):
    find_claw_img = (sct_img_np).astype(np.uint8)[:,:,:3]

    lower_grey = np.array([130, 130, 130])
    upper_grey = np.array([255, 255, 255])
    mask = cv2.inRange(find_claw_img, lower_grey, upper_grey)
    res = cv2.bitwise_and(find_claw_img, find_claw_img, mask=mask)


    points = cv2.findNonZero(mask)
    print(points[0][0][0])

    return 1

    

def claw_loop():
    error = 2
    x,y = centers[0]

    while True:
        sct_img = cv2.imread("find_claw_img.png")

        sct_img_np = np.array(sct_img)

        claw_loc = find_claw(sct_img_np)

        if(claw_loc == None):
            claw_loc = -1

        print(claw_loc)

        if x < claw_loc + error and x > claw_loc - error :
            pyautogui.press('space') 
            print("OVER") 
            return

centers = [(1000, 1000)]
claw_loop()