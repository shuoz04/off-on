import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Camera:
    def __init__(self, capture: Union[int, str] = 0):
        self.cap = cv2.VideoCapture(capture)
        # self.cap.set(3, 1920)
        # self.cap.set(4, 1080)
        self.M = None  # 透视变换矩阵
        self.background = None
        self.vessel = None
        for _ in range(100):
            _, _ = self.cap.read()
            time.sleep(0.05)

    def perspective_initialize(self):
        ret, frame = self.cap.read()
        # plt.imshow(frame)
        # plt.title('请依次点击模型的左上角、左下角、右下角、右上角')
        # point2, point1, point3, point4 = plt.ginput(4, -1)
        # plt.close()
        # print(point2, point1, point3, point4)
        point2, point1, point3, point4 = (65, 19), (71, 467), (521, 463), (515, 10)
        self.M = cv2.getPerspectiveTransform(np.float32([point2, point1, point3, point4]),
                                             np.float32([(0, 0), (0, 300), (300, 300), (300, 0)]))
        self.background = cv2.warpPerspective(frame, self.M, (300, 300))[10:-10, 10:-10]  # shape=[280, 280, 3]
        cv2.imwrite('1.png', self.background)
        self.get_vessel(self.background)
        ax = plt.subplot(2, 1, 1)
        ax.imshow(self.background)
        ax = plt.subplot(2, 1, 2)
        ax.imshow(self.vessel)
        plt.title('background and vessel')
        plt.show()
        return

    def get_vessel(self, fig):
        # plt.imshow(fig)
        # plt.show()
        vessel = (fig[:, :, 2] > 30) + (fig[:, :, 1] < 100)
        # vessel = fig[:, :, 2] < 0.95 * cv2.blur(fig[:, :, 2], (40, 40))
        # vessel = vessel + 0

        # mask for screw
        mask = np.ones_like(vessel)
        mask[0:10, 0:10] = 0
        mask[-10:, 0:10] = 0
        mask[0:10, -10:] = 0
        mask[-10:, -10:] = 0
        mask[90:100, 135:145] = 0
        mask[165:175, 180:190] = 0
        mask[210:225, 105:115] = 0

        vessel = mask*vessel
        vessel = vessel.astype(np.uint8)
        vessel *= 255
        # vessel = cv2.medianBlur(vessel.astype(np.uint8), 5)
        # kernel = np.ones((5, 5), dtype=np.uint8)
        # vessel = cv2.morphologyEx(vessel, cv2.MORPH_CLOSE, kernel)
        # plt.imshow(vessel)
        # plt.show()
        self.vessel = vessel

    def get_figure(self):
        ret, frame = self.cap.read()
        if self.M is None:
            fig = frame
        else:
            fig = cv2.warpPerspective(frame, self.M, (300, 300))[10:-10, 10:-10]  # shape=[280, 280, 3]

        plt.imshow(fig)
        # # points = fig.reshape((-1, 3))
        # # fig = plt.figure(dpi=128, figsize=(8, 8))
        # # ax = fig.add_subplot(111, projection='3d')
        # # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        plt.show()
        #
        # hsv = cv2.cvtColor(fig, cv2.COLOR_BGR2HSV)
        # plt.imshow(hsv)
        # plt.show()
        # lower_black = np.array([0, 0, 0])
        # upper_black = np.array([30, 255, 80])
        # mask = cv2.inRange(hsv, lower_black, upper_black)
        #
        # # Bitwise-AND mask and original image
        # res = cv2.bitwise_and(fig, fig, mask=mask)
        # plt.imshow(mask)
        # plt.show()

        # 导丝检测
        delta = fig.astype(np.int16) - self.background.astype(np.int16)
        gw = (delta[:, :, 0] + delta[:, :, -1]) < -60
        gw = gw + 0
        kernel = np.ones((7, 7), dtype=np.uint8)
        gw = cv2.morphologyEx(gw.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        gw = gw.astype(np.uint8)
        plt.imshow(gw)
        plt.show()
        # gw = cv2.cvtColor(fig, cv2.COLOR_BGR2GRAY) - cv2.cvtColor(self.init_img, cv2.COLOR_BGR2GRAY)

        # 结果可视化
        # plt.clf()
        # ax = plt.subplot(2, 3, 1)
        # ax.imshow(fig)
        # plt.axis('off')
        # ax = plt.subplot(2, 3, 2)
        # ax.imshow(img2)
        # plt.axis('off')
        # ax = plt.subplot(2, 3, 3)
        # plt.imshow(img3)
        # # ax.imshow(cv2.cvtColor(fig, cv2.COLOR_BGR2GRAY))
        # plt.axis('off')
        # ax = plt.subplot(2, 3, 4)
        # plt.imshow(img4)
        # plt.axis('off')
        # ax = plt.subplot(2, 3, 5)
        # plt.imshow(gw)
        # plt.axis('off')
        # ax = plt.subplot(2, 3, 6)
        # plt.imshow(img6)
        # plt.axis('off')
        # plt.show()

        # return np.concatenate((np.expand_dims(self.vessel, axis=2), np.expand_dims(gw, axis=2)), axis=2)  # [280, 280, 2]

        result = fig.copy()
        # 轮廓
        contour = cv2.Canny(self.background, 20, 60) / 255
        contour = contour.astype('uint8')
        kernel = np.ones((3, 3), dtype='uint8')
        contour = cv2.dilate(contour, kernel)
        # ax = plt.subplot(2, 2, 3)
        # ax.set_title(u'轮廓')
        # for i in range(3):
        #     result[:, :, i] *= (1 - contour)

        for i in range(3):
            result[:, :, i] *= (1 - gw)
        return result


if __name__ == '__main__':
    import time
    x = Camera(0)
    x.perspective_initialize()
    # while True:
    #     x.get_figure()
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # input('Press any key to start!')
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (280, 280))
    # for i in range(500):
    #     out.write(x.get_figure())
    #     time.sleep(0.03)
