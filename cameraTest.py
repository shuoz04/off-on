import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage import morphology
import torch


# dif = torch.load('differ.pt')
# plt.imshow(dif)
# plt.show()
#
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
for _ in range(50):
    cap.read()
    time.sleep(0.05)
# point2, point1, point3, point4 = (420, 34), (417, 1043), (1428, 1057), (1435, 30)
# point2, point1, point3, point4 = (394, 35), (390, 1048), (1400, 1060), (1408, 32)
# point2, point1, point3, point4 = (410, 35), (417, 1044), (1432, 1051), (1425, 27)
# point2, point1, point3, point4 = (571, 185), (579, 888), (1288, 887), (1282, 174)
# point2, point1, point3, point4 = (403, 38), (412, 1038), (1432, 1043), (1418, 18)
point2, point1, point3, point4 = (521, 124), (531, 954), (1365, 956), (1356, 112)

M = cv2.getPerspectiveTransform(np.float32([point2, point1, point3, point4]),
                                np.float32([(0, 0), (0, 600), (600, 600), (600, 0)]))

ret, frame = cap.read()
# plt.imshow(frame)
# plt.show()
# frame = cv2.imread('1.jpg')
background = cv2.warpPerspective(frame, M, (600, 600))[20:-20, 20:-20]
# cv2.imwrite('0.jpg', background)
plt.imshow(frame)
plt.show()
# background = cv2.imread('0.jpg')
plt.imshow(background)
plt.show()

# vessel = background[:, :, 2] / background[:, :, 1] > 0.28
vessel = (background[:, :, 1] - background[:, :, 2]) < 90

mask = np.ones_like(vessel)
# mask[:, 0:25] = 0
# mask[0:30, 0:30] = 0
# mask[-30:, 0:30] = 0
# mask[0:30, -30:] = 0
# mask[-30:, -30:] = 0
# mask[410:450, 140:180] = 0
# mask[380:420, 320:360] = 0
# mask[230:270, 280:320] = 0
mask[0:30, 0:30] = 0
mask[-30:, 0:30] = 0
mask[0:30, -30:] = 0
mask[-30:, -30:] = 0
mask[400:440, 140:180] = 0
mask[230:270, 285:325] = 0
mask[375:415, 320:360] = 0
plt.imshow(mask)
plt.show()

vessel = mask * vessel
vessel = vessel.astype(np.uint8)
plt.imshow(vessel)
plt.show()
# img0 = cv2.imread('0.jpg')
# img1 = cv2.imread('1.jpg')
# img2 = cv2.imread('2.jpg')
# # img3 = cv2.imread('3.jpg')
# # img4 = cv2.imread('4.jpg')
# #
# # ax = plt.subplot(3, 1, 1)
# # ax.imshow(img0)
# # ax = plt.subplot(3, 1, 2)
# # ax.imshow(img1)
# # ax = plt.subplot(3, 1, 3)
# # ax.imshow(img1.astype(float) - img0.astype(float))
# # plt.show()
# gw = (img2.astype(float)[:, :, 1] / img0.astype(float)[:, :, 1]) < 0.8
# gw = gw.astype(np.uint8)
# kernel = np.ones((5, 5), dtype=np.uint8)
# gw = cv2.morphologyEx(gw, cv2.MORPH_CLOSE, kernel)
# plt.imshow(gw)
# # skeleton = morphology.skeletonize(gw)
# # gw = cv2.erode(gw.astype(np.uint8), np.ones((3, 3)))
# plt.show()
# init0 = torch.load('camera_init.pt')
# init1 = torch.load('camera_init1.pt')
# plt.imshow(init0['vessel'] > init1['vessel'] )
# plt.show()
