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

while True:
    sct_img = sct.grab(bounding_box)
    sct_img_np = np.array(sct_img)

    find_claw_img = (sct_img_np[0:50, 0:1325]).astype(np.uint8)[:,:,:3]
    w, h = template.shape[:-1]

    res = cv2.matchTemplate(find_claw_img, template, cv2.TM_CCOEFF_NORMED)
    threshold = .8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):  # Switch columns and rows
        cv2.rectangle(sct_img_np, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imshow('screen', sct_img_np)



    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows() 
        break

pyautogui.press('space')    