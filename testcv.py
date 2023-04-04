import cv2
import numpy as np


img_rgb = cv2.imread("test_1.png")
sct_img_np = np.array(img_rgb)
template = cv2.imread('claw.png')

find_claw_img = (sct_img_np[0:50, 0:1325]).astype(np.uint8)[:,:,:3]
w, h = template.shape[:-1]

res = cv2.matchTemplate(find_claw_img, template, cv2.TM_CCOEFF_NORMED)
threshold = .8
loc = np.where(res >= threshold)
print(sum(v[0] for v in loc) / float(len(loc)))
for pt in zip(*loc[::-1]):  # Switch columns and rows
    cv2.rectangle(sct_img_np, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imwrite('result.png', sct_img_np)