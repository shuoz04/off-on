import time
import torch
import cv2
from PyQt5.QtCore import QThread
import numpy as np
from skimage.measure import block_reduce
import paras
import virtualMaster
from cameraThread import detect_ratio, camera_log_path
import os


action_set = [
    # (-10, -10), (-10, -5), (-10, 0), (-10, 5), (-10, 10),
    # (-10, -5), (-10, 5),
    (-5, -10), (-5, -5), (-5, 0), (-5, 5), (-5, 10),
    # (0, -10), (0, -5), (0, 5), (0, 10),
    (5, -10), (5, -5), (5, 0), (5, 5), (5, 10),
    # (10, -10), (10, -5), (10, 0), (10, 5), (10, 10)
    # (10, -5), (10, 5)
]

class basicThread(QThread):
    """
    带reset、step、reward的通用进程类
    """
    def __init__(self,
                 target: int,
                 targets: list,
                 obs: np.ndarray,
                 origin_fig: np.ndarray,
                 vessel_skeleton: np.ndarray,
                 adv_range: int = 10,
                 rot_range: int = 20,
                 stack: bool = False,
                 ):
        super().__init__()
        self.advRange = adv_range
        self.rotRange = rot_range

        self.stack = stack

        self.obs = obs  # shape=[140, 140, 2]
        self.origin_fig = origin_fig
        self.last_obs = None
        self.last_action = None
        # assert obs.shape[0] == 140
        self.remote = np.zeros_like(obs[:, :, 0])
        obsSize = self.obs.shape

        # obs尺寸缩小为原来的1/detect_ratio，所以targets的坐标也要等比例缩小
        targets = targets.copy()
        for i in range(len(targets)):
            targets[i] = (int(round(targets[i][0] / detect_ratio, 0)), int(round(targets[i][1] / detect_ratio, 0)))

        range_size = 3  # 每个分支末端的检测区域大小
        # targets
        self.target = target
        # 导丝远端到达该区域则认为已经到达目标区域
        self.targetRange = np.zeros_like(self.obs[:, :, 0])
        self.targetRange[max(0, targets[target][0] - range_size):min(obsSize[0], targets[target][0] + range_size),
        max(0, targets[target][1] - range_size):min(obsSize[1], targets[target][1] + range_size)] = 1

        # 导丝远端到达该区域则认为已经到达出目标外的血管分叉底部
        self.borderRange = np.zeros_like(self.obs[:, :, 0])
        for i in range(len(targets)):
            if i != target:
                self.borderRange[max(0, targets[i][0] - range_size):min(obsSize[0], targets[i][0] + range_size),
                max(0, targets[i][1] - range_size):min(obsSize[1], targets[i][1] + range_size)] = 1

        points = obs[-1, :, 0].astype(int)
        delta = points[1:] - points[:-1]
        start, end = np.where(delta == 1)[0], np.where(delta == -1)[0]
        assert len(start) == 1
        assert len(end) == 1
        self.origin = (obsSize[0] - 1, int((start + end) / 2))

        self.vessel_skeleton = block_reduce(vessel_skeleton, block_size=(detect_ratio, detect_ratio), func=np.max)

        self.dis_map = None
        self.origin_dis_map = None
        self.dis = None
        self.origin_dis = None
        self.env_step = None

    def run(self):
        pass

    def reset(self):
        """
        reset步骤如下：
        1. 停止所有运动
        2. 回撤导丝直到消失到视野中，发送信号（初始化背景）
        3. 将导丝送至指定位置（导丝远端距离下侧边界5-10像素）
        4. 发送随机时长、随机方向的选择命令，随机化初始角度
        5. 松开导丝，旋转电机复位（发送位置模式为0的指令），重新夹紧导丝
        6. 设置目标
        :return:
        """
        # 停止所有运动
        virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'], 0)
        time.sleep(0.5)
        virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 0)
        #
        # # 回撤导丝直到消失到视野中，发送信号（初始化背景）
        # while True:
        #     if self.obs[-20:, :, 1].sum() > 0:
        #         virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'],
        #                                                             -5)
        #         time.sleep(0.1)
        #     else:
        #         virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 0)
        #         time.sleep(0.1)
        #         break
        #
        # time.sleep(0.5)

        # 将导丝送至指定位置（导丝远端距离下侧边界5-10像素）
        while True:
            if self.obs[:-10, :, 1].sum() > 0:
                virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'],
                                                                    -5)
                time.sleep(0.1)
            elif self.obs[-10:-5, :, 1].sum() == 0:
                virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 5)
                time.sleep(0.1)
            else:
                virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 0)
                time.sleep(0.1)
                break
            # if self.obs[:, :, 1].sum() == 0:
            #     virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 0)
            #     time.sleep(0.2)
            #     break
            # else:
            #     virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'],
            #                                                         -5)
            #     time.sleep(0.2)

        # 发送随机时长、随机方向的选择命令，随机化初始角度
        random_rot = np.random.rand() * 3
        if np.random.randint(2) == 1:
            virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'], 20)
        else:
            virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'], -20)
        time.sleep(random_rot)
        virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'], 0)

        virtualMaster.sendGWRemove()
        time.sleep(1)
        virtualMaster.sendTargetModeOfOperation_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'],
                                                                   'PPM')
        time.sleep(1)
        virtualMaster.sendTargetPosition_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'], 0)
        time.sleep(5)
        virtualMaster.sendGWInstall()
        time.sleep(1)
        virtualMaster.sendTargetModeOfOperation_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'],
                                                                   'PVM')
        time.sleep(1)
        virtualMaster.sendTargetModeOfOperation_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'],
                                                                   'PVM')
        time.sleep(1)
        self.env_step = 0
        self.dis = self.getRemoteDis()
        self.origin_dis = self.getRemoteOriginDis()
        self.last_obs = self.obs
        self.last_action = np.array([10])
        if self.stack:
            obs = (self.last_obs, self.obs, self.last_action)
        else:
            obs = self.obs
        # obs = np.concatenate((self.obs, self.last_obs[:, :, 1:]), axis=2)
        return obs

    def step(self, action: int):
        # assert action.size == 2, f'Illegal Action Error: action size must be 2 but found {action.size}'
        # action = action.reshape(-1)
        # assert -1 <= action[0] <= 1 and -1 <= action[1] <= 1, \
        #     f'Illegal Action Error: each action dimension must be between -1 and 1 but got ({action[0]}, {action[1]})'
        #
        # adv = round(action[0] * self.advRange)
        # rot = round(action[1] * self.rotRange)
        self.last_obs = self.obs
        action = int(action)
        adv = action_set[action][0]
        rot = action_set[action][1] * 3
        virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], adv)
        virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'], rot)
        time.sleep(0.5)
        virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 0)
        virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'], 0)
        time.sleep(0.2)
        if self.stack:
            next_obs = (self.last_obs, self.obs, self.last_action)
        else:
            next_obs = self.obs
        # next_obs = self.obs
        # next_obs = np.concatenate((self.obs, self.last_obs[:, :, 1:]), axis=2)
        self.env_step += 1
        reward = self.reward
        self.last_action = np.array([action])
        if reward[0] > 0 or reward[2] < 0 or self.checkEdge() or self.env_step >= 200:
            # 终止条件：1. 到达目标位置；2. 接触力超过安全阈值；3. 到达边界；4. 到达时长限制
            done = True
        else:
            done = False
        info = {}
        return next_obs, reward, done, info

    @property
    def reward(self):
        new_dis = self.getRemoteDis()
        new_origin_dis = self.getRemoteOriginDis()
        if new_origin_dis < 48:
            currentLimit = 140
        else:
            currentLimit = 170
        if virtualMaster.ActualAdvCurrentOfGw > currentLimit:
            safeReward = -100
            sparseReward, denseReward = 0, 0
        else:
            safeReward = 0

            if (self.obs[:, :, 1] * self.targetRange).any() > 0:
                sparseReward = 40. #400.
            else:
                sparseReward = 0.

            if np.sum(self.obs[:, :, 1]) == 0:  # if exceed vessel
                denseReward = -20
            elif new_dis == -1:  # if in wrong branch
                if self.dis != -1:  # last step is in correct branch, that is, entering the wrong branch
                    denseReward = -10
                else:  # if keeping in wrong branch
                    denseReward = 0
            else:
                if self.dis != -1:
                    denseReward = (self.dis - new_dis)
                else:  # if return to correct branch
                    denseReward = 10

            # if abs(reward) > 50:
            #     new_dis = self.dis
            #     reward = 0
            self.dis = new_dis
            self.origin_dis = new_origin_dis
        rewardSum = sparseReward + denseReward + safeReward
        return sparseReward, denseReward, safeReward, rewardSum

    @property
    def sparseReward(self):
        if (self.obs[:, :, 1] * self.targetRange).any() > 0:
            return 400.
        else:
            return 0.

    @property
    def denseReward(self):
        new_dis = self.getRemoteDis()
        if new_dis == -1:  # if in wrong branch
            if self.dis != -1:  # last step is in correct branch, that is, entering the wrong branch
                reward = -10
            else:  # if keeping in wrong branch
                reward = 0
        else:
            if self.dis != -1:
                reward = (self.dis - new_dis)
            else:  # if return to correct branch
                reward = 10

        # if abs(reward) > 50:
        #     new_dis = self.dis
        #     reward = 0
        self.dis = new_dis
        return reward

    def getRemoteDis(self):
        if self.remote is not None:
            remote = self.remote * self.obs[:, :, 0]
            assert np.sum(remote) <= 1
            if np.sum(remote) == 1:
                index = np.where(remote == 1)
                new_dis = self.dis_map[index[0][0], index[1][0]]
            # elif np.sum(remote) > 1:
            # index = np.where(remote == 1)
            # origin_dis = np.array([self.origin_dis_map[index[0][i], index[1][i]] for i in range(len(index[0]))])
            # new_dis = self.dis_map[index[0][np.argmax(origin_dis)], index[1][np.argmax(origin_dis)]]
            else:
                new_dis = self.dis
        else:
            new_dis = self.dis
        if new_dis > 400:
            new_dis = self.dis
        return new_dis

    def getRemoteOriginDis(self):
        if self.remote is not None:
            remote = self.remote * self.obs[:, :, 0]
            assert np.sum(remote) <= 1
            if np.sum(remote) == 1:
                index = np.where(remote == 1)
                new_dis = self.origin_dis_map[index[0][0], index[1][0]]
            # elif np.sum(remote) > 1:
            # index = np.where(remote == 1)
            # origin_dis = np.array([self.origin_dis_map[index[0][i], index[1][i]] for i in range(len(index[0]))])
            # new_dis = self.dis_map[index[0][np.argmax(origin_dis)], index[1][np.argmax(origin_dis)]]
            else:
                new_dis = self.origin_dis
        else:
            new_dis = self.origin_dis
        if new_dis > 400:
            new_dis = self.origin_dis
        return new_dis

    def checkEdge(self):
        if (self.obs[:, :, 1] * self.borderRange).any() > 0:
            return True
        elif self.obs[:, :, 1].sum() == 0:
            return True
        else:
            return False

    def getDisMap(self):
        obsSize = self.obs.shape
        disMap = np.ones_like(self.obs[:, :, 0], dtype=int) * 500
        disMap[(self.vessel_skeleton * self.targetRange).astype(bool)] = 0
        prefixY, prefixX = np.meshgrid(range(obsSize[0]), range(obsSize[1]))
        prefixX[(self.vessel_skeleton * self.targetRange).astype(bool)] = -1
        prefixY[(self.vessel_skeleton * self.targetRange).astype(bool)] = -1
        trans = [np.array([0, -1, 1, 0, 0, -1, -1, 1, 1]), np.array([0, 0, 0, -1, 1, -1, 1, -1, 1])]
        while True:
            borderMap = cv2.copyMakeBorder(disMap, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=1000)

            new_dis_map = np.min(np.stack([disMap,
                                           borderMap[:-2, 1:-1] + 1,
                                           borderMap[2:, 1:-1] + 1,
                                           borderMap[1:-1, :-2] + 1,
                                           borderMap[1:-1, 2:] + 1,
                                           borderMap[:-2, :-2] + 1,
                                           borderMap[:-2, 2:] + 1,
                                           borderMap[2:, :-2] + 1,
                                           borderMap[2:, 2:] + 1,
                                           ], axis=2), axis=-1)
            new_dis_map[(1 - self.vessel_skeleton).astype(bool)] = disMap[(1 - self.vessel_skeleton).astype(bool)]

            argminMap = np.argmin(np.stack([disMap,
                                            borderMap[:-2, 1:-1] + 1,
                                            borderMap[2:, 1:-1] + 1,
                                            borderMap[1:-1, :-2] + 1,
                                            borderMap[1:-1, 2:] + 1,
                                            borderMap[:-2, :-2] + 1,
                                            borderMap[:-2, 2:] + 1,
                                            borderMap[2:, :-2] + 1,
                                            borderMap[2:, 2:] + 1,
                                            ], axis=2), axis=-1)
            deltaPrefixX = trans[0][argminMap]
            deltaPrefixY = trans[1][argminMap]
            prefixX[self.vessel_skeleton.astype(bool)] += deltaPrefixX[self.vessel_skeleton.astype(bool)]
            prefixY[self.vessel_skeleton.astype(bool)] += deltaPrefixY[self.vessel_skeleton.astype(bool)]

            if (disMap == new_dis_map).all():
                break
            disMap = new_dis_map
        originY = np.argmax(self.vessel_skeleton[-1, :])
        originX = self.vessel_skeleton.shape[0] - 1
        pathMask = np.zeros_like(self.vessel_skeleton)
        pathMask[originX, originY] = 1
        pathMask[(self.vessel_skeleton * self.targetRange).astype(bool)] = 1
        while True:
            if prefixX[originX, originY] == -1:
                break
            else:
                originX, originY = prefixX[originX, originY], prefixY[originX, originY]
                pathMask[originX, originY] = 1
        torch.save(pathMask, os.path.join(camera_log_path, 'path.pt'))
        pathMask = cv2.filter2D(pathMask.astype(np.uint8), -1, np.ones((3, 3))) > 0
        # torch.save(disMap, 'camera_log/skeletonDis1.pt')
        disMap[(self.vessel_skeleton * (1 - pathMask)).astype(bool)] = -1
        # torch.save(disMap, 'camera_log/skeletonDis2.pt')
        self.dis_map = disMap

        skeletonPoints = []
        skeletonDis = []
        for i in range(obsSize[0]):
            for j in range(obsSize[1]):
                if self.vessel_skeleton[i, j]:
                    skeletonPoints.append((i, j))
                    skeletonDis.append(self.dis_map[i, j])
        for i in range(obsSize[0]):
            for j in range(obsSize[1]):
                if self.obs[i, j, 0] > 0 and self.vessel_skeleton[i, j] == 0:
                    dis2skeleton = [np.linalg.norm((point[0] - i, point[1] - j)) for point in skeletonPoints]
                    pointIndex = np.argmin(dis2skeleton)
                    point = skeletonPoints[pointIndex]
                    self.dis_map[i, j] = self.dis_map[
                        point[0], point[1]]  # + np.linalg.norm((point[0] - i, point[1] - j))
        torch.save(self.dis_map, os.path.join(camera_log_path, 'dis.pt'))

        disMap = np.ones_like(self.obs[:, :, 0]) * 500
        disMap[self.origin[0], self.origin[1]] = 0
        flag = True
        while flag:
            new_dis_map = np.min(np.stack([disMap,
                                           np.concatenate([np.ones_like(disMap[0:1, :]) * 1000, disMap[:-1, :]],
                                                          axis=0) + 1,
                                           np.concatenate([disMap[1:, :], np.ones_like(disMap[0:1, :]) * 1000],
                                                          axis=0) + 1,
                                           np.concatenate([np.ones_like(disMap[:, 0:1]) * 1000, disMap[:, :-1]],
                                                          axis=1) + 1,
                                           np.concatenate([disMap[:, 1:], np.ones_like(disMap[:, 0:1]) * 1000],
                                                          axis=1) + 1,
                                           ], axis=2), axis=-1)
            new_dis_map[~self.obs[:, :, 0].astype(bool)] = disMap[~self.obs[:, :, 0].astype(bool)]
            if (disMap == new_dis_map).all():
                flag = False

            disMap = new_dis_map
        self.origin_dis_map = disMap
        torch.save(self.origin_dis_map, os.path.join(camera_log_path, 'origin_dis.pt'))

        self.dis = self.dis_map[self.origin[0], self.origin[1]]
