import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import torch
from skimage import morphology
from skimage.measure import block_reduce
import os


# point2, point1, point3, point4 = (420, 34), (417, 1043), (1428, 1057), (1435, 30)
# point2, point1, point3, point4 = (406, 40), (408, 1056), (1418, 1060), (1419, 34)
# point2, point1, point3, point4 = (410, 35), (417, 1044), (1432, 1051), (1425, 27)
# point2, point1, point3, point4 = (571, 185), (579, 888), (1288, 887), (1282, 174)
# point2, point1, point3, point4 = (403, 38), (412, 1038), (1432, 1043), (1418, 18)

point2, point1, point3, point4 = (521, 124), (531, 954), (1365, 956), (1356, 112)


fig_size = 600
detect_ratio = 4
edge_distance = 20
M = cv2.getPerspectiveTransform(np.float32([point2, point1, point3, point4]),
                                np.float32([(0, 0), (0, fig_size), (fig_size, fig_size), (fig_size, 0)]))

# mask for screw
mask = np.ones((fig_size - 2*edge_distance, fig_size - 2*edge_distance))
# mask[0:20, 0:20] = 0
# mask[-20:, 0:20] = 0
# mask[0:20, -20:] = 0
# mask[-20:, -20:] = 0
# mask[170:200, 260:290] = 0
# mask[320:360, 360:390] = 0
# mask[410:450, 200:240] = 0
mask[0:30, 0:30] = 0
mask[-30:, 0:30] = 0
mask[0:30, -30:] = 0
mask[-30:, -30:] = 0
# mask for model 1
# mask[410:450, 200:240] = 0
# mask[320:360, 350:390] = 0
# mask[170:210, 260:290] = 0
# mask for model 2
mask[400:440, 140:180] = 0
mask[230:270, 285:325] = 0
mask[375:415, 320:360] = 0

camera_log_path = 'camera_log_20231205'

class initCameraThread(QThread):
    signal = pyqtSignal(cv2.VideoCapture, np.ndarray, np.ndarray, np.ndarray, list)

    def __init__(self):
        super().__init__()

    def run(self):
        """
        1. get camera capture; 2. get background; 3. get vessel and vessel skeleton; 4. get targets
        :return:
        """
        cap = cv2.VideoCapture(0)
        cap.set(3, 1920)
        cap.set(4, 1080)
        for _ in range(60):
            _, _ = cap.read()
            time.sleep(0.05)

        imgs = []
        for _ in range(5):
            ret, frame = cap.read()
            background = cv2.warpPerspective(frame, M, (fig_size, fig_size))[edge_distance:-edge_distance,
                         edge_distance:-edge_distance]
            imgs.append(background)
            time.sleep(0.5)
        background = np.array(imgs).mean(axis=0).astype(np.uint8)
        vessel, vessel_skeleton = self.getVessel(background)
        if not os.path.exists(camera_log_path):
            os.makedirs(camera_log_path)
        camera_init = torch.load(os.path.join(camera_log_path, 'camera_init.pt'))
        torch.save(camera_init['vessel'] != vessel, os.path.join(camera_log_path, 'differ.pt'))
        torch.save(vessel, os.path.join(camera_log_path, 'vessel.pt'))
        print((camera_init['vessel'] != vessel).sum())
        if (camera_init['vessel'] != vessel).sum() < 8000:
            vessel = camera_init['vessel']
            vessel_skeleton = camera_init['vessel_skeleton']
            targets = camera_init['targets']
            # print('1')
        else:
            print('Camera inits differently!!!')
            print((camera_init['vessel'] != vessel).sum())
            targets = self.getTargets(vessel_skeleton)

        # targets = self.getTargets(vessel_skeleton)
        # torch.save({'vessel': vessel, 'vessel_skeleton': vessel_skeleton, 'targets': targets},
        #            os.path.join(camera_log_path, 'camera_init.pt'))
        self.signal.emit(cap, background, vessel, vessel_skeleton, targets)

    def getVessel(self, fig):
        """
        从background中分割血管部分
        :return: 血管, 血管中线
        """
        # vessel = fig[:, :, 2] / fig[:, :, 1] > 0.28

        vessel = (fig[:, :, 1] - fig[:, :, 2]) < 90
        vessel = mask * vessel
        vessel = vessel.astype(np.uint8)

        kernel = np.ones((5, 5), dtype=np.uint8)
        vessel_skeleton = morphology.skeletonize(
            cv2.copyMakeBorder(cv2.erode(vessel, kernel), 5, 5, 5, 5, borderType=cv2.BORDER_REPLICATE))[5:-5, 5:-5]

        # plt.imshow(vessel_skeleton)
        # plt.show()
        # vessel *= 255
        # vessel = cv2.medianBlur(vessel.astype(np.uint8), 5)
        # kernel = np.ones((3, 3), dtype=np.uint8)
        # vessel = cv2.morphologyEx(vessel, cv2.MORPH_CLOSE, kernel)
        return vessel, vessel_skeleton.astype(np.uint8)

    def getTargets(self, vessel_skeleton):
        """
        从self.vessel中分割血管通路的终点作为目标
        :return: 目标位置
        """
        targets = []

        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dst = cv2.filter2D(vessel_skeleton, -1, kernel, borderType=cv2.BORDER_CONSTANT) * vessel_skeleton
        index = np.where(dst == (kernel_size + 1) / 2)
        for i in range(len(index[0])):
            targets.append((index[0][i], index[1][i]))
        targets = targets[:-1]  # 去掉起始位置
        print(targets)
        return targets


class cameraThread(QThread):
    signal = pyqtSignal((np.ndarray, np.ndarray, np.ndarray))

    def __init__(self, cap, background, vessel):
        super().__init__()
        self.cap = cap
        self.background = background
        torch.save(background, 'camera_log_new/background.pt')
        self.fig = self.background
        self.vessel = vessel

    def run(self):
        points = self.vessel[-1, :].astype(int)
        delta = points[1:] - points[:-1]
        start, end = np.where(delta == 1)[0], np.where(delta == -1)[0]
        assert len(start) == 1
        assert len(end) == 1
        origin = (self.vessel.shape[0] - 1, int((start + end) / 2))
        dis_map = np.ones_like(self.vessel) * 1500
        dis_map[origin[0], origin[1]] = 0
        while True:
            new_dis_map = np.min(np.stack([dis_map,
                                           np.concatenate([np.ones_like(dis_map[0:1, :]) * 1000, dis_map[:-1, :]],
                                                          axis=0) + 1,
                                           np.concatenate([dis_map[1:, :], np.ones_like(dis_map[0:1, :]) * 1000],
                                                          axis=0) + 1,
                                           np.concatenate([np.ones_like(dis_map[:, 0:1]) * 1000, dis_map[:, :-1]],
                                                          axis=1) + 1,
                                           np.concatenate([dis_map[:, 1:], np.ones_like(dis_map[:, 0:1]) * 1000],
                                                          axis=1) + 1,
                                           ], axis=2), axis=-1)
            new_dis_map[~self.vessel.astype(bool)] = dis_map[~self.vessel.astype(bool)]
            if (dis_map == new_dis_map).all():
                break
            dis_map = new_dis_map
        # plt.imshow(dis_map)
        # plt.show()

        origin_dis_map = dis_map

        # last_remote = None

        while True:
            ret, frame = self.cap.read()
            if ret:
                self.fig = cv2.warpPerspective(frame, M, (fig_size, fig_size))[edge_distance:-edge_distance,
                           edge_distance:-edge_distance]

                # 导丝检测
                # gw = (self.fig.astype(float)[:, :, 1] / self.background.astype(float)[:, :, 1]) < 0.75
                gw = (self.fig.astype(float)[:, :, 1] / self.background.astype(float)[:, :, 1]) < 0.8
                gw = gw.astype(np.uint8) * self.vessel
                # kernel = np.ones((31, 31), dtype=np.uint8)
                kernel = np.ones((5, 5), dtype=np.uint8)
                # gw = cv2.morphologyEx(gw, cv2.MORPH_CLOSE, kernel)
                gw = cv2.dilate(gw, kernel)
                gw = morphology.skeletonize(cv2.copyMakeBorder(gw, 30, 30, 30, 30, borderType=cv2.BORDER_REPLICATE))[
                     30:-30, 30:-30]
                gw = gw.astype(np.uint8) * self.vessel

                # 导丝远端检测
                # hsv = cv2.cvtColor(fig, cv2.COLOR_BGR2HSV)
                # lower_blue = np.array([156, 43, 46])
                # upper_blue = np.array([190, 255, 255])
                # mask = cv2.inRange(hsv, lower_blue, upper_blue)
                # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                # skeleton = morphology.skeletonize(gw) * self.vessel
                kernel_size = 3
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                dst = cv2.filter2D(gw.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT) * gw
                index = np.where(dst == (kernel_size + 1) / 2)
                if len(index[0]) == 0:
                    remote = np.zeros(0)
                elif len(index[0]) == 1:
                    remote = np.zeros_like(gw)
                    remote[index[0][0], index[1][0]] = 1
                    # if last_remote is not None:
                    #     if abs(last_remote[0] - index[0][0]) + abs(last_remote[1] - index[1][0]) > 100:
                    #         print(1)
                    # last_remote = (index[0][0], index[1][0])

                else:
                    remote = np.zeros_like(gw)
                    origin_dis = np.array([origin_dis_map[index[0][i], index[1][i]] for i in range(len(index[0]))])
                    remote[index[0][np.argmax(origin_dis)], index[1][np.argmax(origin_dis)]] = 1
                    # if last_remote is not None:
                    #     if abs(last_remote[0] - index[0][np.argmax(origin_dis)]) + abs(last_remote[1] - index[1][np.argmax(origin_dis)]) > 100:
                    #         print(1)
                    # last_remote = (index[0][np.argmax(origin_dis)], index[1][np.argmax(origin_dis)])

                self.signal.emit(self.fig, gw, remote)

                # time.sleep(0.02)

    def initBackground(self):
        """
        trainThread的槽函数：在每次reset的时候重新设置背景
        :return:
        """
        imgs = []
        for _ in range(5):
            ret, frame = self.cap.read()
            background = cv2.warpPerspective(frame, M, (fig_size, fig_size))[edge_distance:-edge_distance,
                         edge_distance:-edge_distance]
            imgs.append(background)
            time.sleep(0.2)
        self.background = np.array(imgs).mean(axis=0).astype(np.uint8)

