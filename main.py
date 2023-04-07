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

def find_cheese(img_rgb):
    img_rgb = (img_rgb).astype(np.uint8)[:,:,:3]

    lower_yellow = np.array([25, 150, 0])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(img_rgb, lower_yellow, upper_yellow)
    res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)


    points = cv2.findNonZero(mask)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Find x,y coordinates of all non-white pixels in original image
    Y, X = np.where(mask==255)
    Z = np.column_stack((X,Y)).astype(np.float32)

    nClusters = 2
    ret,label,center=cv2.kmeans(Z,nClusters,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


    return center

def find_claw(sct_img_np):
    find_claw_img = (sct_img_np[0:50, 0:1325]).astype(np.uint8)[:,:,:3]
    w, h = template.shape[:-1]

    res = cv2.matchTemplate(find_claw_img, template, cv2.TM_CCOEFF_NORMED)



    threshold = .8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):  # Switch columns and rows
        return pt[0] + 10


running = True

error = 10

sct_img = sct.grab(bounding_box)

sct_img_np = np.array(sct_img)

centers = find_cheese(sct_img_np)
 

while running:
    
    sct_img = sct.grab(bounding_box)

    sct_img_np = np.array(sct_img)

    claw_loc = find_claw(sct_img_np)

    for x,y in centers:
        print("____")
        print(x)
        print(claw_loc)
        print("____")

        if x < claw_loc + error and x > claw_loc - error:
            print("OVER") 
            running = False


pyautogui.press('space')    
print(click)




 