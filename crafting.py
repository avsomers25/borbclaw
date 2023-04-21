import pyautogui
import time

import cv2
import numpy as np

from mss import mss

sct = mss() 
 
bounding_box = {'top': 690, 'left': 320, 'width': 1325, 'height': 10 }    

template = cv2.imread('claw.png')

filename = sct.shot()

lower_white = np.array([250, 250, 250])
upper_white = np.array([255, 255, 255])

def find_line_2(sct_img_np):
    find_claw_img = (sct_img_np).astype(np.uint8)[:,:,:3]

    mask = cv2.inRange(find_claw_img, lower_white, upper_white)
    res = cv2.bitwise_and(find_claw_img, find_claw_img, mask=mask)

    points = cv2.findNonZero(mask)
                           
    try:
        return points[0][0][0]
    except TypeError:
        return -1



print("wait") 
time.sleep(5)
print("go")
pyautogui.press('space')    

while True:


    sct_img = sct.grab(bounding_box)

    sct_img_np = np.array(sct_img)

    line_loc = find_line_2(sct_img_np)

    if(line_loc > 650 and line_loc < 675): 
        pyautogui.press('space')     
        time.sleep(2)
        pyautogui.press('space')       


