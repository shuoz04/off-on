import json
import os
import sys
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from torch.distributions import Categorical
import torch.nn as nn
from gui import Ui_Form
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import virtualMaster
import paras
import msvcrt
import ray
import torch
import argparse
from retrying import retry
from skimage.measure import block_reduce
from network_architecture import ALIXEncoder1, ALIXEncoder3, ParameterizedReg, \
    LocalSignalMixing, StackPolicyNetwork, PolicyNetwork
from masterThread import masterThread
from cameraThread import cameraThread, initCameraThread, detect_ratio
from basicThread import basicThread
import DSAC_learner
import torch.nn.functional as F

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

action_set = [
    # (-10, -10), (-10, -5), (-10, 0), (-10, 5), (-10, 10),
    # (-10, -5), (-10, 5),
    (-5, -10), (-5, -5), (-5, 0), (-5, 5), (-5, 10),
    # (0, -10), (0, -5), (0, 5), (0, 10),
    (5, -10), (5, -5), (5, 0), (5, 5), (5, 10),
    # (10, -10), (10, -5), (10, 0), (10, 5), (10, 10)
    # (10, -5), (10, 5)
]


class RLEnvGUI(QtWidgets.QMainWindow):
    def __init__(self, target=0):
        super(RLEnvGUI, self).__init__()
        self.gui = Ui_Form()
        self.gui.setupUi(self)

        self.gui.startTrainButton.setEnabled(False)
        self.gui.stopTrainButton.setEnabled(False)
        self.gui.testButton.setEnabled(False)
        self.gui.startRuleBasedDeliveryButton.setEnabled(False)
        self.gui.setTargetButton.setEnabled(False)
        self.gui.setStepButton.setEnabled(False)
        self.gui.target.setEnabled(False)
        self.gui.max_step.setEnabled(False)

        self.gui.action1.hide()
        self.gui.action2.hide()
        self.gui.done.hide()
        self.gui.contact.hide()
        self.gui.step.hide()
        self.gui.action1_label.hide()
        self.gui.action2_label.hide()
        self.gui.done_label.hide()
        self.gui.contact_label.hide()
        self.gui.step_label.hide()
        self.gui.init_label.hide()

        # 按钮槽函数
        self.gui.initCameraButton.clicked.connect(self.initCameraClick)
        self.gui.startMasterButton.clicked.connect(self.startMasterClick)
        self.gui.startTrainButton.clicked.connect(self.startTrainClick)
        self.gui.stopTrainButton.clicked.connect(self.stopTrainClick)
        self.gui.testButton.clicked.connect(self.testClick)
        self.gui.startRuleBasedDeliveryButton.clicked.connect(self.RuleBasedDeliveryClick)
        self.gui.setTargetButton.clicked.connect(self.setTargetClick)
        self.gui.setStepButton.clicked.connect(self.setStepClick)

        # 图像相关
        self.cap = None  # 相机
        self.frame = None  # 当前相机图像
        self.modelRange = None  # 血管模型图像
        self.background = None  # 背景
        self.vessel = None  # 血管区域
        self.vessel_skeleton = None  # 血管骨架
        self.gw = None  # 导丝区域
        self.remote = None  # 导丝远端

        # 机器人相关
        self.connectFlag = False  # 是否与从端连接的标志位

        # 强化学习相关
        self.targets = []
        self.target = target
        self.gui.target.setText(f'{self.target}')
        self.max_step = 1e3
        self.gui.max_step.setText('%d' % self.max_step)
        # 进程
        self.initCameraThread = None  # 摄像机初始化进程
        self.cameraThread = None  # 摄像机读取进程
        self.masterThread = None  # 虚拟主端连接进程
        self.trainThread = None  # 训练进程
        self.manualThread = None  # 手动控制进程
        self.videoThread = None  # 录像进程
        self.autoDeliveryThread = None
        self.testThread = None

    # ------------------------GUI槽函数---------------------------------------------#
    def initCameraClick(self):
        """
        initCameraButton.clicked的槽函数：开始initCameraThread
        :return:
        """
        if self.initCameraThread is None:
            self.gui.initCameraButton.setEnabled(False)
            self.initCameraThread = initCameraThread()
            self.initCameraThread.signal.connect(self.initCamera)
            self.initCameraThread.start()

    def startMasterClick(self):
        """
        startMasterButton.clicked的槽函数：开始masterThread
        :return:
        """
        if self.masterThread is None:
            self.gui.startMasterButton.setEnabled(False)
            self.masterThread = masterThread()
            self.masterThread.signal.connect(self.setConnectFlag)
            self.masterThread.start()

    def startTrainClick(self):
        """
        startTrainButton.clicked的槽函数：开始trainThread
        :return:
        """
        self.gui.startTrainButton.setEnabled(False)
        self.gui.testButton.setEnabled(False)
        self.gui.stopTrainButton.setEnabled(True)
        if self.trainThread is None:
            # self.trainThread = trainThread1(self.target, self.targets.copy(), self.observation, self.modelRange, self.vessel_skeleton)
            self.trainThread = trainThreadDSACAE(self.target, self.targets.copy(), self.observation, self.modelRange,
                                                 self.vessel_skeleton)

            self.trainThread.signal.connect(self.cameraThread.initBackground)
            # self.trainThread = rewardTestThread(self.targets[self.target], self.getState())
        self.trainThread.start()

    def stopTrainClick(self):
        self.gui.startTrainButton.setEnabled(True)
        self.gui.testButton.setEnabled(True)
        self.gui.stopTrainButton.setEnabled(False)
        self.trainThread.terminate()
        virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 0)
        virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'], 0)
        # self.trainThread = None

    def testClick(self):
        if self.gui.testButton.text() == 'Test':
            file_name, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "*.pt;*.pkl")
            # file_name = ['D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0313-10000-cql\\beta0.1-seed1\\200k\\200k.pt',
            #              'D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0313-10000-cql\\beta0.2-seed1\\200k\\200k.pt',
            #              'D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0313-10000-cql\\beta0.3-seed1\\200k\\200k.pt',
            #              'D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0313-10000-cql\\beta0.4-seed1\\200k\\200k.pt',
            #              'D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0313-10000-cql\\beta0.5-seed1\\200k\\200k.pt',
            #              'D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0313-10000-cql\\beta0.1-seed2\\200k\\200k.pt',
            #              'D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0313-10000-cql\\beta0.2-seed2\\200k\\200k.pt',
            #              'D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0313-10000-cql\\beta0.3-seed2\\200k\\200k.pt',
            #              'D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0313-10000-cql\\beta0.4-seed2\\200k\\200k.pt',
            #              'D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0313-10000-cql\\beta0.5-seed2\\200k\\200k.pt',
            #              ]
            # file_name = [
            #              'D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0702-10000-CQL\\1\\200k\\200k.pt',
            #              'D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0702-10000-CQL\\2\\200k\\200k.pt',
            #              'D:\\LH\\程序\\自主递送\\offline+online\\trained model\\0702-10000-CQL\\3\\200k\\200k.pt',
            #              ]
            # file_name = ['D:\\LH\程序\\自主递送\\offline+online\\trained model\\model-based\\20231031-net5-kl0.000005-feature256']
            if type(file_name) != list:
                file_name = [file_name]
            if self.testThread is None:
                # self.testThread = testThread(self.target, self.targets.copy(), self.observation, self.modelRange,
                #                              self.vessel_skeleton, model_path=[file_name])
                # self.testThread = testThread(self.target, self.targets.copy(), self.observation, self.modelRange,
                #                              self.vessel_skeleton, model_path=file_name)
                self.testThread = modelBasedTestThread(self.target, self.targets.copy(), self.observation,
                                                       self.modelRange,
                                                       self.vessel_skeleton, model_path=file_name)
                # self.testThread = modelBasedTestThreadwoSharedEncoder(self.target, self.targets.copy(), self.observation,
                #                                        self.modelRange, self.vessel_skeleton, model_path=file_name)
                # self.testThread = BCQtestThread(self.target, self.targets.copy(), self.observation, self.modelRange,
                #                              self.vessel_skeleton, model_path=file_name)
                # self.testThread.signal.connect(self.cameraThread.initBackground)
                self.testThread.signal.connect(self.saveTestVIdeo)
            else:
                self.testThread.model_path = file_name
                self.testThread.log_path = None
            self.testThread.start()
            # t = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
            # self.videoThread = videoThread(self.modelRange, 'test_%s.mp4' % t)
            # self.videoThread.start()
            self.gui.testButton.setText('Stop Test')
        else:
            self.testThread.terminate()
            # self.videoThread.flag = False
            virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 0)
            virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'], 0)
            self.gui.testButton.setText('Test')
        return

    def RuleBasedDeliveryClick(self):
        if self.gui.startRuleBasedDeliveryButton.text() == 'Rule-Based':
            if self.observation is None:
                print('Camera is not initialsed!')
            elif virtualMaster.SlaveState != paras.MS_STAT_E['MS_STAT_OK']:
                print('Robot is not connected!')
            else:
                self.gui.setTargetButton.setEnabled(False)
                self.gui.setStepButton.setEnabled(False)
                self.gui.target.setEnabled(False)
                self.gui.max_step.setEnabled(False)
                # if self.autoDeliveryThread is None:
                self.autoDeliveryThread = ruleBasedDeliveryThread(self.target, self.targets.copy(), self.observation,
                                                                  self.modelRange, self.vessel_skeleton,
                                                                  step=self.max_step, random_ratio=0.15)
                self.autoDeliveryThread.signal.connect(self.showMessage)
                self.gui.action1.hide()
                self.gui.action2.hide()
                self.gui.done.hide()
                self.gui.contact.hide()
                self.gui.step.hide()
                self.gui.action1_label.hide()
                self.gui.action2_label.hide()
                self.gui.done_label.hide()
                self.gui.contact_label.hide()
                self.gui.step_label.hide()
                self.gui.init_label.show()
                self.autoDeliveryThread.start()
                # t = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
                # self.videoThread = videoThread(self.modelRange, 'test_%s.mp4' % t)
                # self.videoThread.start()
                self.gui.startRuleBasedDeliveryButton.setText('Stop')

        else:
            self.autoDeliveryThread.terminate()
            self.gui.setTargetButton.setEnabled(True)
            self.gui.target.setEnabled(True)
            self.gui.setStepButton.setEnabled(True)
            self.gui.max_step.setEnabled(True)
            virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 0)
            virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'], 0)
            self.gui.startRuleBasedDeliveryButton.setText('Rule-Based')
            if self.videoThread is not None:
                self.saveTestVIdeo('', False)
        return

    def setTargetClick(self):
        target = int(self.gui.target.text())
        if 0 <= target < len(self.targets):
            self.target = target
        else:
            QMessageBox.warning(self, '目标范围错误', f'指定的目标范围为0~{len(self.targets) - 1}',
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes)
        # print(type(self.gui.target.text()), self.gui.target.text())
        return

    def setStepClick(self):
        max_step = int(self.gui.max_step.text())
        if 0 < max_step <= 1e8:
            self.max_step = max_step
        else:
            QMessageBox.warning(self, '采集步数范围错误', f'指定的采集步数范围为1~{1e8}',
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes)
        # print(type(self.gui.target.text()), self.gui.target.text())
        return

    # ------------------------Thread槽函数---------------------------------------------#
    def initCamera(self, *args):
        """
        initCameraThread的槽函数：从摄像头中初始化background和vessel，然后开始cameraThread
        :param args: 摄像头(class cv2.capture)
        :return:
        """
        self.cap, self.background, self.vessel, self.vessel_skeleton, self.targets = args
        # plt.imshow(self.vessel)
        # plt.show()
        self.showVessel()
        # self.gui.initCameraButton.setEnabled(True)
        self.gui.setTargetButton.setEnabled(True)
        self.gui.target.setEnabled(True)
        self.gui.setStepButton.setEnabled(True)
        self.gui.max_step.setEnabled(True)
        if self.connectFlag:
            self.gui.startTrainButton.setEnabled(True)
            self.gui.testButton.setEnabled(True)
            self.gui.startRuleBasedDeliveryButton.setEnabled(True)

        self.cameraThread = cameraThread(self.cap, self.background, self.vessel)
        self.cameraThread.signal.connect(self.camera)
        self.cameraThread.start()
        self.gui.initCameraButton.setEnabled(True)

    def camera(self, *args):
        """
        cameraThread的槽函数：更新并展示模型图像和导丝区域
        :param args: 当前模型的图像和导丝区域
        :return:
        """
        self.modelRange, self.gw, remote = args
        if len(remote.shape) > 1:
            self.remote = remote
        # print(sum(self.gw))
        if self.trainThread is not None:
            self.trainThread.obs = self.observation
            if self.remote is not None:
                self.trainThread.remote = block_reduce(self.remote, block_size=(detect_ratio, detect_ratio),
                                                       func=np.max)
        if self.autoDeliveryThread is not None:
            self.autoDeliveryThread.obs = self.observation
            self.autoDeliveryThread.origin_fig = self.modelRange
            if self.remote is not None:
                self.autoDeliveryThread.remote = block_reduce(self.remote, block_size=(detect_ratio, detect_ratio),
                                                              func=np.max)
        if self.videoThread is not None:
            self.videoThread.obs = self.modelRange
        if self.testThread is not None:
            self.testThread.obs = self.observation
            if self.remote is not None:
                self.testThread.remote = block_reduce(self.remote, block_size=(detect_ratio, detect_ratio),
                                                      func=np.max)
        self.showModelRange()
        self.showGW()
        self.showRemote()

    def setConnectFlag(self, flag):
        """
        masterThread的槽函数：设置与从端的连接状态
        :param flag: 与从端的连接状态
        :return:
        """
        self.connectFlag = flag
        if self.cameraThread is not None:
            self.gui.startTrainButton.setEnabled(True)
            self.gui.testButton.setEnabled(True)
            self.gui.startRuleBasedDeliveryButton.setEnabled(True)

    def saveTestVIdeo(self, file_name, flag):
        if flag:
            self.videoThread = videoThread(self.modelRange, file_name)
            self.videoThread.start()
        else:
            self.videoThread.flag = False

    def showMessage(self, step, action, reward, done, current, initial, finish, reset, episodes):
        """
        collectThread的槽函数，实时显示信息
        :param step:
        :param action:
        :param reward:
        :param done:
        :param current:
        :param initial:
        :param finish:
        :param reset:
        :return:
        """
        if finish:
            self.RuleBasedDeliveryClick()
        elif reset:
            self.saveTestVIdeo(os.path.join('rule-based delivery', '20231205', str(episodes) + '.mp4'), True)
            if initial:
                self.gui.action1.show()
                self.gui.action2.show()
                self.gui.done.show()
                self.gui.contact.show()
                self.gui.step.show()
                self.gui.action1_label.show()
                self.gui.action2_label.show()
                self.gui.done_label.show()
                self.gui.contact_label.show()
                self.gui.step_label.show()
                self.gui.init_label.hide()
        else:
            self.gui.action1.setText(str(action_set[action][0]))
            self.gui.action2.setText(str(action_set[action][1]))
            if done:
                self.gui.done.setText('是')
                self.saveTestVIdeo(os.path.join('rule-based delivery', '20231031', str(episodes) + '.mp4'), False)
            else:
                self.gui.done.setText('否')
            self.gui.contact.setText(str(current))
            print(current)
            self.gui.step.setText(str(step))

    # ------------------------图像预处理---------------------------------------------#
    @property
    def observation(self):
        """
        获得当前观测值(vessel和guidewire图像)
        """
        vessel = block_reduce(self.vessel, block_size=(detect_ratio, detect_ratio), func=np.max)
        gw = block_reduce(self.gw, block_size=(detect_ratio, detect_ratio), func=np.max)
        obs = np.concatenate((np.expand_dims(vessel, axis=2), np.expand_dims(gw, axis=2)), axis=2)
        return obs

    # ------------------------可视化函数---------------------------------------------#
    def showVessel(self):
        vesselView = cv2.cvtColor(cv2.resize(self.vessel * 255, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
        vesselView = QtGui.QImage(vesselView.data, vesselView.shape[1], vesselView.shape[0], QtGui.QImage.Format_RGB888)
        self.gui.vesselViewLabel.setPixmap(QtGui.QPixmap.fromImage(vesselView))
        return

    def showGW(self):
        GWView = cv2.cvtColor(cv2.resize(self.gw * 255, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
        GWView = QtGui.QImage(GWView.data, GWView.shape[1], GWView.shape[0], QtGui.QImage.Format_RGB888)
        self.gui.guidewireViewLabel.setPixmap(QtGui.QPixmap.fromImage(GWView))
        return

    def showRemote(self):
        if self.remote is not None:
            kernel = np.ones((9, 9), np.uint8)
            remote = cv2.resize(cv2.filter2D(self.remote, -1, kernel), None, fx=0.5, fy=0.5)
            remoteView = cv2.cvtColor(remote, cv2.COLOR_BGR2RGB) > 0
            remoteView = remoteView.astype(np.uint8) * 255
            remoteView = QtGui.QImage(remoteView.data, remoteView.shape[1], remoteView.shape[0],
                                      QtGui.QImage.Format_RGB888)
            self.gui.remoteViewLabel.setPixmap(QtGui.QPixmap.fromImage(remoteView))
        return

    def showModelRange(self):
        # view = block_reduce(self.modelRange, block_size=(2, 2, 1), func=np.mean).astype(np.uint8)
        # if self.targets is not None:
        #     target = self.targets[self.target]
        #     view[target[0] - 5:target[0] + 5, target[1] - 5:target[1] + 5, 0] = 0
        #     view[target[0] - 5:target[0] + 5, target[1] - 5:target[1] + 5, 1] = 0
        #     view[target[0] - 5:target[0] + 5, target[1] - 5:target[1] + 5, 2] = 255
        modelRange = self.modelRange.copy()
        obsSize = modelRange.shape
        range_size = 9
        if len(self.targets) > 0:
            modelRange[max(0, self.targets[self.target][0] - range_size):min(obsSize[0],
                                                                             self.targets[self.target][0] + range_size),
            max(0, self.targets[self.target][1] - range_size):min(obsSize[1],
                                                                  self.targets[self.target][1] + range_size), 0] = 0
            modelRange[max(0, self.targets[self.target][0] - range_size):min(obsSize[0],
                                                                             self.targets[self.target][0] + range_size),
            max(0, self.targets[self.target][1] - range_size):min(obsSize[1],
                                                                  self.targets[self.target][1] + range_size), 1] = 0
            modelRange[max(0, self.targets[self.target][0] - range_size):min(obsSize[0],
                                                                             self.targets[self.target][0] + range_size),
            max(0, self.targets[self.target][1] - range_size):min(obsSize[1],
                                                                  self.targets[self.target][1] + range_size), 2] = 255
        view = cv2.cvtColor(cv2.resize(modelRange, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
        view = QtGui.QImage(view.data, view.shape[1], view.shape[0], QtGui.QImage.Format_RGB888)
        self.gui.cameraViewLabel.setPixmap(QtGui.QPixmap.fromImage(view))
        return


# ------------------------进程定义---------------------------------------------#
# 采集基于规则的数据
class ruleBasedDeliveryThread(basicThread):
    # signal contains step number, action, reward, done, current, initialize finish, collection finish, reset, eposide
    signal = pyqtSignal(int, int, float, bool, int, bool, bool, bool, int)

    def __init__(self, target, targets, obs, origin_fig, vessel_skeleton, adv_range=10, rot_range=20, stack=False,
                 step=1e3, random_ratio=0.2):
        super().__init__(target, targets, obs, origin_fig, vessel_skeleton, adv_range, rot_range, stack)
        self.max_step = step
        self.random_ratio = random_ratio

    def run(self):
        save_path = os.path.join('rule-based delivery', '20231205')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.dis_map is None:
            self.getDisMap()
        self.env_step = 0

        # 机器人初始化
        virtualMaster.sendGWInstall()
        virtualMaster.sendTargetModeOfOperation_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'],
                                                                   'PVM')
        virtualMaster.sendTargetModeOfOperation_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'],
                                                                   'PVM')
        obs = self.reset()
        buffer = []
        env_step = 0
        true_path_flag = True  # whether the guidewire is on the true path
        back_step = 0
        episodes = 0
        self.signal.emit(env_step, -1, 0, False, 0, True, False, True, episodes)
        while env_step < self.max_step:
            if np.random.uniform() < self.random_ratio:
                action = np.random.randint(len(action_set), size=1)
            else:
                if true_path_flag:
                    if back_step == 0:
                        action = np.random.randint(5, size=1) + 5
                    else:
                        action = np.random.randint(5, size=1)
                        back_step -= 1
                else:
                    action = np.random.randint(5, size=1)
            next_obs, reward, done, _ = self.step(action)
            buffer.append(
                (self.origin_fig, obs, action, reward, next_obs, done, virtualMaster.ActualAdvCurrentOfGw, self.remote))
            # print(f'step:{env_step} | action:{action} | reward:{reward} | done: {done}')

            if true_path_flag and reward[1] == -10:  # 从正确路径进入错误路径
                true_path_flag = False
            elif not true_path_flag and reward[1] == 10:  # 从错误路径回到正确路径
                true_path_flag = True
                back_step = np.random.randint(2, 4)
            self.signal.emit(env_step, int(action), reward[-1], done, virtualMaster.ActualAdvCurrentOfGw, False, False,
                             False, episodes)
            if done:
                torch.save(buffer, os.path.join(save_path, f'{episodes}.pt'))
                obs = self.reset()
                episodes += 1
                self.signal.emit(env_step, -1, 0, False, 0, False, False, True, episodes)
                true_path_flag = True
                back_step = 0
                buffer = []
            else:
                obs = next_obs
            env_step += 1
        if len(buffer) > 0:
            torch.save(buffer, os.path.join(save_path, f'{episodes + 1}.pt'))
        self.signal.emit(0, -1, 0, False, 0, False, True, False, episodes)


# 离线训练
class trainThread1(basicThread):
    signal = pyqtSignal()

    def __init__(self,
                 target: int,
                 targets: list,
                 obs: np.ndarray,
                 origin_fig: np.ndarray,
                 vessel_skeleton: np.ndarray,
                 adv_range: int = 10,
                 rot_range: int = 20,
                 stack: bool = False,
                 # model_path: str = 'offlineRL/100k.pt',
                 # log_path: str = 'bc_entropy1.3_test'):
                 log_path: str = 'test',
                 args_path: str = 'trained model/0130-11/args.json'):
        super().__init__(target, targets, obs, origin_fig, vessel_skeleton, adv_range, rot_range, stack)
        self.log_path = log_path
        self.args_path = args_path
        self.last_obs = None
        self.last_action = None
        self.noise = 0.2

    def run(self):

        device = torch.device('cpu')
        with open(self.args_path, 'r') as f:
            args = f.read()
            args = json.loads(args)
        channals = 4
        aug = ParameterizedReg(aug=LocalSignalMixing(pad=2, fixed_batch=True),
                               parameter_init=0.5, param_grad_fn='alix_param_grad',
                               param_grad_fn_args=[3, 0.535, 1e-20])
        if 'encoder' in args.keys():
            encoder = eval(args['encoder'])((140, 140, channals), aug=aug).to(device)
        else:
            encoder = ALIXEncoder1((140, 140, channals), aug=aug).to(device)

        act_net = StackPolicyNetwork(feature_dim=encoder.feature_dim,
                                     action_dim=len(action_set),
                                     hidden_dim=args['actor_hidden_dim'])
        # 加载模型参数是在learner中
        # payload = torch.load(self.model_path[0], map_location='cpu')
        # encoder.load_state_dict(payload['encoder'])
        # if 'policy' in payload.keys():
        #     act_net.load_state_dict(payload['policy'])
        # elif 'actor' in payload.keys():
        #     act_net.load_state_dict(payload['actor'])
        # print(f'successfully load offline model "{self.model_path[0]}"!')
        done = True
        if self.dis_map is None:
            self.getDisMap()
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.env_step = 0  # 环境的时间不大于100秒（1000步）

        # 机器人初始化
        virtualMaster.sendGWInstall()
        virtualMaster.sendTargetModeOfOperation_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'],
                                                                   'PVM')
        virtualMaster.sendTargetModeOfOperation_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'],
                                                                   'PVM')

        parser = argparse.ArgumentParser()

        # ray arguments
        parser.add_argument("--address", type=str, help="the address of ray head node",
                            default='ray://172.18.22.9:10001')
        parser.add_argument("--namespace", type=str, help="the name of node", default='actor')
        parser.add_argument("--dataset-namespace", type=str, help="the namespace of dataset node", default='dataset')
        parser.add_argument("--dataset-name", type=str, help="the name of replay buffer", default='dataset-node')
        parser.add_argument("--learner-namespace", type=str, help="the namespace of learner node", default='learner')
        parser.add_argument("--learner-name", type=str, help="the name of learner", default='learner-node')
        parser.add_argument("--min-learn-act-ratio", type=float, help="the min ratio of learn step to act step",
                            default=0.2)

        # RL arguments
        # parser.add_argument("--hidden-size", type=int, help="the hidden size of policy network", default=512)
        parser.add_argument("--step", type=int, help="the number of total transition", default=1e6)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args = parser.parse_args()
        print('node actor: ------  ' + args.namespace + ' node connect to ray cluster  ------')
        if not ray.is_initialized():
            ray.init(address=args.address, namespace=args.namespace)

        print('node actor: ------  initialize actor  ------')
        # env = gym.make(args.env)

        # print(list(act_net.parameters()))
        learner = get_remote_class(name=args.learner_name, namespace=args.learner_namespace)
        # learner_step = 0
        learner_step = copy_remote_paras2(act_net, encoder, learner)
        # print(list(act_net.parameters()))
        print('node actor: ------  connect to dataset  ------')
        dataset = get_remote_class(name=args.dataset_name, namespace=args.dataset_namespace)
        ray.get(dataset.reset.remote())  # 重置buffer
        t = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
        log_file = os.path.join(self.log_path, f'actor_log_{t}.txt')
        with open(log_file, 'w') as f:
            f.write('time,step,sparse_return,dense_return,safe_return,backward\n')
        obs = self.reset()  # last_obs, obs, last_action
        last_obs = obs.copy()
        last_action = np.array([10])

        step = 0
        # episode_returns = []
        # episode_return = 0
        episode_sparse_return = 0
        episode_dense_return = 0
        episode_safe_return = 0
        episode_backward = 0
        episode_sparse_returns = []
        episode_dense_returns = []
        episode_safe_returns = []
        episode_backwards = []
        while step < args.step:
            # choose action
            if np.random.uniform() < self.noise:
                action = np.random.randint(len(action_set), size=1)
            else:
                feature = encoder(np.concatenate([last_obs, obs], axis=-1))
                action = act_net.get_action(feature, last_action)

                if action < 5:
                    episode_backward += 1
            # action = np.array([-0.5, -0.7])
            next_obs, reward, done, _ = self.step(action)
            print(step, action_set[int(action)][0], action_set[int(action)][1], reward,
                  virtualMaster.ActualAdvCurrentOfGw, virtualMaster.ActualAdvTorqueOfGw)
            # episode_return += reward[-1]
            episode_sparse_return += reward[0]
            episode_dense_return += reward[1]
            episode_safe_return += reward[2] / 5
            if last_action == 10:
                ray.get(dataset.push.remote(obs, action, reward[1] + reward[2] / 5, next_obs, done))
            else:
                ray.get(dataset.push.remote(None, action, reward[1] + reward[2] / 5, next_obs, done))
            step += 1

            if done:
                if len(episode_sparse_returns) >= 5:
                    episode_sparse_returns.pop(0)
                    episode_dense_returns.pop(0)
                    episode_safe_returns.pop(0)
                    episode_backwards.pop(0)
                episode_sparse_returns.append(episode_sparse_return)
                episode_dense_returns.append(episode_dense_return)
                episode_safe_returns.append(episode_safe_return)
                episode_backwards.append(episode_backward)
                print('sparse_return: %d | dense_return: %d | safe_return: %d | backward: %d' %
                      (episode_sparse_return, episode_dense_return, episode_safe_return, episode_backward))
                episode_sparse_return = 0
                episode_dense_return = 0
                episode_safe_return = 0
                episode_backward = 0
                obs = self.reset()
                last_obs = obs.copy()
                last_action = np.array([10])
                # 每次重置环境时检查actor是否运行太快
                while step > 2000 and ray.get(dataset.learn_act_ratio.remote()) < args.min_learn_act_ratio:
                    time.sleep(1)
            else:
                last_obs = obs.copy()
                last_action = action
                obs = next_obs

            if step % 1000 == 0:
                mean_sparse_return = np.array(episode_sparse_returns).mean()
                mean_dense_return = np.array(episode_dense_returns).mean()
                mean_safe_return = np.array(episode_safe_returns).mean()
                mean_backward = np.array(episode_backwards).mean()
                time_str = time.strftime('%dsY-%m-%d %H:%M:%S', time.localtime(time.time()))
                print(f'node actor: {time_str} actor  {int(step / 1000)}k transitions have been sampled. ' +
                      'mean_sparse_return: %.2f | mean_dense_return: %.2f | mean_safe_return: %.2f | mean_backward: %.2f' %
                      (mean_sparse_return, mean_dense_return, mean_safe_return, mean_backward))
                with open(log_file, 'a') as f:
                    f.write(f"{time_str},{int(step / 1000)}k,%.2f,%.2f,%.2f,%.2f\n" % (
                        mean_sparse_return, mean_dense_return, mean_safe_return, mean_backward))
                # learner_step = copy_remote_paras2(act_net, encoder, learner)
            if step % 50 == 0:
                learner_step = copy_remote_paras2(act_net, encoder, learner)
            if step % 100 == 0:
                self.noise *= 0.95


class trainThreadDSACAE(basicThread):
    signal = pyqtSignal()

    def __init__(self,
                 target: int,
                 targets: list,
                 obs: np.ndarray,
                 origin_fig: np.ndarray,
                 vessel_skeleton: np.ndarray,
                 adv_range: int = 10,
                 rot_range: int = 20,
                 stack: bool = False,
                 # model_path: str = 'offlineRL/100k.pt',
                 # log_path: str = 'bc_entropy1.3_test'):
                 log_path: str = 'test',
                 args_path: str = 'trained model/0130-11/args.json'):
        super().__init__(target, targets, obs, origin_fig, vessel_skeleton, adv_range, rot_range, stack)
        self.log_path = log_path
        self.args_path = args_path
        self.last_obs = None
        self.last_action = None

    def run(self):

        device = torch.device('cpu')

        act_net = DSAC_learner.PolicyNetwork((140, 140, 2), 10, 512).to(device)
        # 加载模型参数是在learner中
        # payload = torch.load(self.model_path[0], map_location='cpu')
        # encoder.load_state_dict(payload['encoder'])
        # if 'policy' in payload.keys():
        #     act_net.load_state_dict(payload['policy'])
        # elif 'actor' in payload.keys():
        #     act_net.load_state_dict(payload['actor'])
        # print(f'successfully load offline model "{self.model_path[0]}"!')
        done = True
        if self.dis_map is None:
            self.getDisMap()
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.env_step = 0  # 环境的时间不大于100秒（1000步）

        # 机器人初始化
        virtualMaster.sendGWInstall()
        virtualMaster.sendTargetModeOfOperation_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'],
                                                                   'PVM')
        virtualMaster.sendTargetModeOfOperation_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'],
                                                                   'PVM')

        parser = argparse.ArgumentParser()

        # ray arguments
        parser.add_argument("--address", type=str, help="the address of ray head node",
                            default='ray://172.18.22.9:10001')
        parser.add_argument("--namespace", type=str, help="the name of node", default='actor')
        parser.add_argument("--dataset-namespace", type=str, help="the namespace of dataset node", default='dataset')
        parser.add_argument("--dataset-name", type=str, help="the name of replay buffer", default='dataset-node')
        parser.add_argument("--learner-namespace", type=str, help="the namespace of learner node", default='learner')
        parser.add_argument("--learner-name", type=str, help="the name of learner", default='learner-node')
        parser.add_argument("--min-learn-act-ratio", type=float, help="the min ratio of learn step to act step",
                            default=0.2)

        # RL arguments
        # parser.add_argument("--hidden-size", type=int, help="the hidden size of policy network", default=512)
        parser.add_argument("--step", type=int, help="the number of total transition", default=1e6)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args = parser.parse_args()
        print('node actor: ------  ' + args.namespace + ' node connect to ray cluster  ------')
        if not ray.is_initialized():
            ray.init(address=args.address, namespace=args.namespace)

        print('node actor: ------  initialize actor  ------')
        # env = gym.make(args.env)

        # print(list(act_net.parameters()))
        learner = get_remote_class(name=args.learner_name, namespace=args.learner_namespace)
        learner_step = copy_remote_paras(act_net, learner)
        # print(list(act_net.parameters()))
        print('node actor: ------  connect to dataset  ------')
        dataset = get_remote_class(name=args.dataset_name, namespace=args.dataset_namespace)
        ray.get(dataset.reset.remote())  # 重置buffer
        t = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
        log_file = os.path.join(self.log_path, f'actor_log_{t}.txt')
        with open(log_file, 'w') as f:
            f.write('time,step,sparse_return,dense_return,safe_return,backward\n')
        obs = self.reset()  # last_obs, obs, last_action

        step = 0
        # episode_returns = []
        # episode_return = 0
        episode_sparse_return = 0
        episode_dense_return = 0
        episode_safe_return = 0
        episode_backward = 0
        episode_sparse_returns = []
        episode_dense_returns = []
        episode_safe_returns = []
        episode_backwards = []
        while step < args.step:
            # choose action
            if step < 2000:
                action = np.random.randint(len(action_set), size=1)
            else:
                action = act_net.get_action(obs)

                if action < 5:
                    episode_backward += 1
            # action = np.array([-0.5, -0.7])
            next_obs, reward, done, _ = self.step(action)
            print(step, action_set[int(action)][0], action_set[int(action)][1], reward,
                  virtualMaster.ActualAdvCurrentOfGw, virtualMaster.ActualAdvTorqueOfGw)
            # episode_return += reward[-1]
            episode_sparse_return += reward[0]
            episode_dense_return += reward[1]
            episode_safe_return += reward[2] / 5
            ray.get(dataset.push.remote(obs, action, reward[1] + reward[2] / 5, next_obs, done))

            step += 1

            if done:
                if len(episode_sparse_returns) >= 5:
                    episode_sparse_returns.pop(0)
                    episode_dense_returns.pop(0)
                    episode_safe_returns.pop(0)
                    episode_backwards.pop(0)
                episode_sparse_returns.append(episode_sparse_return)
                episode_dense_returns.append(episode_dense_return)
                episode_safe_returns.append(episode_safe_return)
                episode_backwards.append(episode_backward)
                print('sparse_return: %d | dense_return: %d | safe_return: %d | backward: %d' %
                      (episode_sparse_return, episode_dense_return, episode_safe_return, episode_backward))
                episode_sparse_return = 0
                episode_dense_return = 0
                episode_safe_return = 0
                episode_backward = 0
                obs = self.reset()
                # 每次重置环境时检查actor是否运行太快
                while step > 2000 and ray.get(dataset.learn_act_ratio.remote()) < args.min_learn_act_ratio:
                    time.sleep(1)
            else:
                obs = next_obs

            if step % 1000 == 0:
                mean_sparse_return = np.array(episode_sparse_returns).mean()
                mean_dense_return = np.array(episode_dense_returns).mean()
                mean_safe_return = np.array(episode_safe_returns).mean()
                mean_backward = np.array(episode_backwards).mean()
                time_str = time.strftime('%dsY-%m-%d %H:%M:%S', time.localtime(time.time()))
                print(f'node actor: {time_str} actor  {int(step / 1000)}k transitions have been sampled. ' +
                      'mean_sparse_return: %.2f | mean_dense_return: %.2f | mean_safe_return: %.2f | mean_backward: %.2f' %
                      (mean_sparse_return, mean_dense_return, mean_safe_return, mean_backward))
                with open(log_file, 'a') as f:
                    f.write(f"{time_str},{int(step / 1000)}k,%.2f,%.2f,%.2f,%.2f\n" % (
                        mean_sparse_return, mean_dense_return, mean_safe_return, mean_backward))
                # learner_step = copy_remote_paras2(act_net, encoder, learner)
            if step % 50 == 0:
                learner_step = copy_remote_paras(act_net, learner)


class rewardTestThread(trainThread1):
    def __init__(self, target, targets, obs, origin_fig, vessel_skeleton, adv_range=10, rot_range=20, stack=True):
        super().__init__(target, targets, obs, origin_fig, vessel_skeleton, adv_range, rot_range, stack)

    def run(self):
        while True:
            print(self.reward)
            time.sleep(0.5)


class manualThread(QThread):
    def __init__(self, adv_range=10, rot_range=50):
        super().__init__()
        self.advRange = adv_range
        self.rotRange = rot_range

    def run(self):
        while True:
            key = msvcrt.getch()
            if key == b'w':
                virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'],
                                                                    int(self.advRange * 0.5))
            elif key == b's':
                virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'],
                                                                    -int(self.advRange * 0.5))
            elif key == b'a':
                virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 0)
                virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'],
                                                                    int(self.rotRange * 0.5))
            elif key == b'd':
                virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 0)
                virtualMaster.sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'],
                                                                    -int(self.rotRange * 0.5))
            time.sleep(0.05)


class videoThread(QThread):
    def __init__(self, obs, videoName='test.mp4'):
        super().__init__()
        self.obs = obs
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(videoName, fourcc, 10.0, (560, 560))
        self.step = 0
        self.flag = True
        print(f'video start:{videoName}')

    def run(self):
        while self.flag:
            if self.writer is not None:
                assert self.obs.shape == (560, 560, 3), print(self.obs.shape)
                self.writer.write(self.obs)
            time.sleep(0.1)
            self.step += 1
        self.writer.release()
        print('video finish')
        return


class testThread(basicThread):
    signal = pyqtSignal(str, bool)

    def __init__(self,
                 target: int,
                 targets: list,
                 obs: np.ndarray,
                 origin_fig: np.ndarray,
                 vessel_skeleton: np.ndarray,
                 adv_range: int = 10,
                 rot_range: int = 20,
                 stack: bool = True,
                 model_path: list = None,
                 # model_path: str = 'offlineRL/100k.pt',
                 # log_path: str = 'bc_entropy1.3_test'):
                 log_path: list = None,
                 test_num: int = 50):
        """
        对训练好的模型进行测试，参数文件args.json位于模型文件的父文件夹下
        :param target:
        :param targets: 所有可能的目标点，不包含起点，不需要缩小
        :param obs: 系统状态（二值图像），已缩小
        :param vessel_skeleton: 血管骨架， 不需要缩小
        :param adv_range:
        :param rot_range:
        :param model_path: 模型存储路径
        :param log_path: 测试结果保存路径
        :param test_num: 每次测试的递送数量
        """
        super().__init__(target, targets, obs, origin_fig, vessel_skeleton, adv_range, rot_range, stack)

        self.model_path = model_path
        self.log_path = log_path
        self.test_num = test_num

    def run(self):
        if self.model_path is None:
            self.model_path = []
            for p in ['offlineRL_stack_test\\stack-3\\entropy1.0-policylr1e-5-warm10000-encoder1-PER-full-1\\']:
                # model_path.append(p + '10k\\10k.pt')
                self.model_path.append(p + '30k\\30k.pt')
                self.model_path.append(p + '50k\\50k.pt')
                self.model_path.append(p + '100k\\100k.pt')
                # model_path.append(p + '200k\\200k.pt')
            # model_path = ['offlineRL_stack_test\\stack\\entropy1.0-policylr1e-5-warm10000-encoder1-PER-full-1\\30k\\30k.pt']
        if self.log_path is None:
            self.log_path = []
            # for p in ['offlineRL_stack_test\\stack-3\\entropy1.0-policylr1e-5-warm10000-encoder1-PER-full-1\\']:
            #     # self.log_path.append(p + '10k')
            #     self.log_path.append(p + '30k')
            #     self.log_path.append(p + '50k')
            #     self.log_path.append(p + '100k')
            # self.log_path.append(p + '200k')
            for p in self.model_path:
                self.log_path.append(os.path.dirname(p))
            # log_path = ['offlineRL_stack_test\\stack\\entropy1.0-policylr1e-5-warm10000-encoder1-PER-full-1\\30k']

        assert len(self.model_path) == len(self.log_path)
        device = torch.device('cpu')
        # encoder = network_architecture.Encoder((140, 140, 2)).to(device)
        aug = ParameterizedReg(aug=LocalSignalMixing(pad=2, fixed_batch=True),
                               parameter_init=0.5, param_grad_fn='alix_param_grad',
                               param_grad_fn_args=[3, 0.535, 1e-20])

        # encoder = encoderExactor((140, 140, 2), 20, 128).to(device)
        if self.dis_map is None:
            self.getDisMap()
        for i in range(len(self.model_path)):
            with open(os.path.join(self.log_path[i], os.pardir, 'args.json'), 'r') as f:
                args = f.read()
                args = json.loads(args)
            if self.stack:
                channals = 4
            else:
                channals = 2
            if 'encoder' in args.keys():
                encoder = eval(args['encoder'])((140, 140, channals), aug=aug).to(device)
            else:
                encoder = ALIXEncoder1((140, 140, channals), aug=aug).to(device)

            if self.stack:
                act_net = StackPolicyNetwork(feature_dim=encoder.feature_dim,
                                             action_dim=len(action_set),
                                             hidden_dim=args['actor_hidden_dim'])
            else:
                act_net = PolicyNetwork(feature_dim=encoder.feature_dim, action_dim=len(action_set),
                                        hidden_dim=64).to(device)

            payload = torch.load(self.model_path[i], map_location='cpu')
            encoder.load_state_dict(payload['encoder'])
            encoder.train(False)
            if 'policy' in payload.keys():
                act_net.load_state_dict(payload['policy'])
            elif 'actor' in payload.keys():
                act_net.load_state_dict(payload['actor'])
            print(f'successfully load model "{self.model_path[i]}"!')
            done = True
            if not os.path.exists(self.log_path[i]):
                os.makedirs(self.log_path[i])
            test_num = 0
            with open(os.path.join(self.log_path[i], f'{test_num}.txt'), 'w') as f:
                f.write('action;reward;done\n')
            obs = self.reset()
            self.signal.emit(os.path.join(self.log_path[i], f'{test_num}.mp4'), True)
            successful_num = 0
            back_num = 0
            back_sum = 0
            reward_sum = 0
            buffer = []
            with torch.no_grad():
                while test_num < self.test_num:
                    if self.stack:
                        feature = encoder(np.concatenate([obs[0], obs[1]], axis=-1))
                        action = act_net.get_action(feature, obs[2])
                    else:
                        feature = encoder(obs)
                        action = act_net.get_action(feature)  # np.ndarray shape = 1
                    if action < 5:
                        back_num += 1
                    next_obs, reward, done, _ = self.step(action)
                    buffer.append(
                        (self.origin_fig, obs, action, reward, next_obs, done, virtualMaster.ActualAdvCurrentOfGw,
                         self.remote))
                    reward_sum += reward[1] + reward[2] / 5
                    with open(os.path.join(self.log_path[i], f'{test_num}.txt'), 'a') as f:
                        f.write(f'{action};{reward};{done}\n')
                    # print(reward, done)
                    if done:
                        if reward[0] > 0:
                            successful_num += 1
                            back_sum += back_num
                        self.signal.emit(os.path.join(self.log_path[i], f'{test_num}.mp4'), False)
                        back_num = 0
                        obs = self.reset()
                        done = False
                        # print(test_num, reward)
                        torch.save(buffer, os.path.relpath(os.path.join(self.log_path[i], f'{test_num}.pt')))
                        buffer = []
                        test_num += 1
                        if test_num != self.test_num:
                            with open(os.path.join(self.log_path[i], f'{test_num}.txt'), 'w') as f:
                                f.write('action;reward;done\n')
                            self.signal.emit(os.path.join(self.log_path[i], f'{test_num}.mp4'), True)
                    else:
                        obs = next_obs
                    # action = np.array([-0.5, -0.7])
            with open(os.path.join(self.log_path[i], os.pardir, os.pardir, 'results.txt'), 'a') as f:
                if successful_num == 0:
                    print(self.model_path[i], f'{successful_num}/{self.test_num}', 0.0, reward_sum / test_num)
                    f.write(f'{self.model_path[i]}, {successful_num}/{self.test_num}, 0.0, {reward_sum / test_num}\n')
                else:
                    print(self.model_path[i], f'{successful_num}/{self.test_num}', back_sum / successful_num,
                          reward_sum / test_num)
                    f.write(
                        f'{self.model_path[i]}, {successful_num}/{self.test_num}, {back_sum / successful_num}, {reward_sum / test_num}\n')


class modelBasedTestThread(basicThread):
    signal = pyqtSignal(str, bool)

    def __init__(self,
                 target: int,
                 targets: list,
                 obs: np.ndarray,
                 origin_fig: np.ndarray,
                 vessel_skeleton: np.ndarray,
                 adv_range: int = 10,
                 rot_range: int = 20,
                 stack: bool = True,
                 model_path: list = None,
                 # model_path: str = 'offlineRL/100k.pt',
                 # log_path: str = 'bc_entropy1.3_test'):
                 test_num: int = 50):
        """
        对训练好的模型进行测试，参数文件args.json位于模型文件的父文件夹下
        :param target:
        :param targets: 所有可能的目标点，不包含起点，不需要缩小
        :param obs: 系统状态（二值图像），已缩小
        :param vessel_skeleton: 血管骨架， 不需要缩小
        :param adv_range:
        :param rot_range:
        :param model_path: 模型存储路径
        :param log_path: 测试结果保存路径
        :param test_num: 每次测试的递送数量
        """
        super().__init__(target, targets, obs, origin_fig, vessel_skeleton, adv_range, rot_range, stack)

        self.model_path = model_path
        self.test_num = test_num

    def run(self):
        device = torch.device('cpu')
        import model_based_offline5
        import model_based_offline6
        if self.dis_map is None:
            self.getDisMap()
        for i in range(len(self.model_path)):
            with open(os.path.join(os.path.dirname(self.model_path[i]), 'args.json'), 'r') as f:
                args = f.read()
                args = json.loads(args)
                args = argparse.Namespace(**args)
            trainer = model_based_offline6.StackTrainer(obs_shape=(140, 140, 4),
                                                        action_dim=10,
                                                        actor_hidden_dim=args.actor_hidden_dim,
                                                        critic_hidden_dim=args.critic_hidden_dim,
                                                        gamma=args.gamma,
                                                        device=device,
                                                        log_alpha=args.log_alpha,
                                                        target_entropy=args.target_entropy,
                                                        soft_q_lr=args.soft_q_lr,
                                                        alpha_lr=args.alpha_lr,
                                                        policy_lr=args.policy_lr,
                                                        env_lr=args.env_lr,
                                                        soft_tau=args.soft_tau,
                                                        PER_flag=args.PER_flag,
                                                        priorities_coefficient=args.PER_coefficient,
                                                        priorities_bias=args.PER_bias,
                                                        kl_weight=args.env_kl_weight,
                                                        feature_dim=args.env_feature_dim)
            payload = torch.load(
                os.path.join(os.path.dirname(self.model_path[i]), os.path.pardir,
                             args.env_save_path, 'best.pt'),
                map_location=torch.device('cpu'))
            trainer.env_model.load_state_dict(payload['env'])
            payload = torch.load(os.path.join(self.model_path[i]),
                                 map_location=torch.device('cpu'))
            trainer.actor.load_state_dict(payload['actor'])
            trainer.soft_q_nets.load_state_dict(payload['critic'])
            trainer.target_soft_q_nets.load_state_dict(payload['target_critic'])
            trainer.log_alpha = payload['log_alpha']
            print(f'successfully load model "{self.model_path[i]}"!')
            done = True
            test_num = 0
            with open(os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.txt'), 'w') as f:
                f.write('action;reward;done\n')
            obs = self.reset()
            self.signal.emit(os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.mp4'), True)
            successful_num = 0
            back_num = 0
            back_sum = 0
            reward_sum = 0
            buffer = []
            with torch.no_grad():
                while test_num < self.test_num:
                    # feature = trainer.env_model.get_feature(torch.cat([obs[0], obs[1]], dim=1))
                    action = trainer.get_action(torch.FloatTensor(np.moveaxis(obs[0], -1, -3)).unsqueeze(0),
                                                torch.FloatTensor(np.moveaxis(obs[1], -1, -3)).unsqueeze(0),
                                                obs[2])
                    if action < 5:
                        back_num += 1
                    next_obs, reward, done, _ = self.step(action)
                    buffer.append(
                        (self.origin_fig, obs, action, reward, next_obs, done, virtualMaster.ActualAdvCurrentOfGw,
                         self.remote))
                    reward_sum += reward[1] + reward[2] / 5
                    with open(os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.txt'), 'a') as f:
                        f.write(f'{action};{reward};{done}\n')
                    # print(reward, done)
                    if done:
                        if reward[0] > 0:
                            successful_num += 1
                            back_sum += back_num
                        self.signal.emit(os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.mp4'), False)
                        back_num = 0
                        obs = self.reset()
                        done = False
                        # print(test_num, reward)
                        torch.save(buffer, os.path.relpath(
                            os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.pt')))
                        buffer = []
                        test_num += 1
                        if test_num != self.test_num:
                            with open(os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.txt'), 'w') as f:
                                f.write('action;reward;done\n')
                            self.signal.emit(os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.mp4'),
                                             True)
                    else:
                        obs = next_obs
                    # action = np.array([-0.5, -0.7])
            with open(os.path.join(os.path.dirname(self.model_path[i]), os.path.pardir, 'results.txt'), 'a') as f:
                if successful_num == 0:
                    print(self.model_path[i], f'{successful_num}/{self.test_num}', 0.0, reward_sum / test_num)
                    f.write(f'{self.model_path[i]}, {successful_num}/{self.test_num}, 0.0, {reward_sum / test_num}\n')
                else:
                    print(self.model_path[i], f'{successful_num}/{self.test_num}', back_sum / successful_num,
                          reward_sum / test_num)
                    f.write(
                        f'{self.model_path[i]}, {successful_num}/{self.test_num}, {back_sum / successful_num}, {reward_sum / test_num}\n')


class modelBasedTestThreadwoSharedEncoder(basicThread):
    signal = pyqtSignal(str, bool)

    def __init__(self,
                 target: int,
                 targets: list,
                 obs: np.ndarray,
                 origin_fig: np.ndarray,
                 vessel_skeleton: np.ndarray,
                 adv_range: int = 10,
                 rot_range: int = 20,
                 stack: bool = True,
                 model_path: list = None,
                 # model_path: str = 'offlineRL/100k.pt',
                 # log_path: str = 'bc_entropy1.3_test'):
                 test_num: int = 50):
        """
        对训练好的模型进行测试，参数文件args.json位于模型文件的父文件夹下
        :param target:
        :param targets: 所有可能的目标点，不包含起点，不需要缩小
        :param obs: 系统状态（二值图像），已缩小
        :param vessel_skeleton: 血管骨架， 不需要缩小
        :param adv_range:
        :param rot_range:
        :param model_path: 模型存储路径
        :param log_path: 测试结果保存路径
        :param test_num: 每次测试的递送数量
        """
        super().__init__(target, targets, obs, origin_fig, vessel_skeleton, adv_range, rot_range, stack)

        self.model_path = model_path
        self.test_num = test_num

    def run(self):
        device = torch.device('cpu')
        import model_based_offline6_wo_shared_encoder
        if self.dis_map is None:
            self.getDisMap()
        for i in range(len(self.model_path)):
            with open(os.path.join(os.path.dirname(self.model_path[i]), 'args.json'), 'r') as f:
                args = f.read()
                args = json.loads(args)
                args = argparse.Namespace(**args)
            trainer = model_based_offline6_wo_shared_encoder.StackTrainer(obs_shape=(140, 140, 4),
                                                                          action_dim=10,
                                                                          actor_hidden_dim=args.actor_hidden_dim,
                                                                          critic_hidden_dim=args.critic_hidden_dim,
                                                                          gamma=args.gamma,
                                                                          device=device,
                                                                          log_alpha=args.log_alpha,
                                                                          target_entropy=args.target_entropy,
                                                                          soft_q_lr=args.soft_q_lr,
                                                                          alpha_lr=args.alpha_lr,
                                                                          policy_lr=args.policy_lr,
                                                                          env_lr=args.env_lr,
                                                                          soft_tau=args.soft_tau,
                                                                          PER_flag=args.PER_flag,
                                                                          priorities_coefficient=args.PER_coefficient,
                                                                          priorities_bias=args.PER_bias,
                                                                          kl_weight=args.env_kl_weight,
                                                                          feature_dim=args.env_feature_dim)
            payload = torch.load(
                os.path.join(os.path.dirname(self.model_path[i]), os.path.pardir,
                             args.env_save_path, 'best.pt'),
                map_location=torch.device('cpu'))
            trainer.env_model.load_state_dict(payload['env'])
            payload = torch.load(os.path.join(self.model_path[i]),
                                 map_location=torch.device('cpu'))
            trainer.actor.load_state_dict(payload['actor'])
            trainer.soft_q_nets.load_state_dict(payload['critic'])
            trainer.target_soft_q_nets.load_state_dict(payload['target_critic'])
            trainer.log_alpha = payload['log_alpha']
            print(f'successfully load model "{self.model_path[i]}"!')
            done = True
            test_num = 0
            with open(os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.txt'), 'w') as f:
                f.write('action;reward;done\n')
            obs = self.reset()
            self.signal.emit(os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.mp4'), True)
            successful_num = 0
            back_num = 0
            back_sum = 0
            reward_sum = 0
            buffer = []
            with torch.no_grad():
                while test_num < self.test_num:
                    # feature = trainer.env_model.get_feature(torch.cat([obs[0], obs[1]], dim=1))
                    action = trainer.get_action(torch.FloatTensor(np.moveaxis(obs[0], -1, -3)).unsqueeze(0),
                                                torch.FloatTensor(np.moveaxis(obs[1], -1, -3)).unsqueeze(0),
                                                obs[2])
                    if action < 5:
                        back_num += 1
                    next_obs, reward, done, _ = self.step(action)
                    buffer.append(
                        (self.origin_fig, obs, action, reward, next_obs, done, virtualMaster.ActualAdvCurrentOfGw,
                         self.remote))
                    reward_sum += reward[1] + reward[2] / 5
                    with open(os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.txt'), 'a') as f:
                        f.write(f'{action};{reward};{done}\n')
                    # print(reward, done)
                    if done:
                        if reward[0] > 0:
                            successful_num += 1
                            back_sum += back_num
                        self.signal.emit(os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.mp4'), False)
                        back_num = 0
                        obs = self.reset()
                        done = False
                        # print(test_num, reward)
                        torch.save(buffer, os.path.relpath(
                            os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.pt')))
                        buffer = []
                        test_num += 1
                        if test_num != self.test_num:
                            with open(os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.txt'), 'w') as f:
                                f.write('action;reward;done\n')
                            self.signal.emit(os.path.join(os.path.dirname(self.model_path[i]), f'{test_num}.mp4'),
                                             True)
                    else:
                        obs = next_obs
                    # action = np.array([-0.5, -0.7])
            with open(os.path.join(os.path.dirname(self.model_path[i]), os.path.pardir, 'results.txt'), 'a') as f:
                if successful_num == 0:
                    print(self.model_path[i], f'{successful_num}/{self.test_num}', 0.0, reward_sum / test_num)
                    f.write(f'{self.model_path[i]}, {successful_num}/{self.test_num}, 0.0, {reward_sum / test_num}\n')
                else:
                    print(self.model_path[i], f'{successful_num}/{self.test_num}', back_sum / successful_num,
                          reward_sum / test_num)
                    f.write(
                        f'{self.model_path[i]}, {successful_num}/{self.test_num}, {back_sum / successful_num}, {reward_sum / test_num}\n')


class BCQtestThread(basicThread):
    signal = pyqtSignal(str, bool)

    def __init__(self,
                 target: int,
                 targets: list,
                 obs: np.ndarray,
                 origin_fig: np.ndarray,
                 vessel_skeleton: np.ndarray,
                 adv_range: int = 10,
                 rot_range: int = 20,
                 stack: bool = True,
                 model_path: list = None,
                 # model_path: str = 'offlineRL/100k.pt',
                 # log_path: str = 'bc_entropy1.3_test'):
                 log_path: list = None,
                 test_num: int = 50):
        """
        对训练好的模型进行测试，参数文件args.json位于模型文件的父文件夹下
        :param target:
        :param targets: 所有可能的目标点，不包含起点，不需要缩小
        :param obs: 系统状态（二值图像），已缩小
        :param vessel_skeleton: 血管骨架， 不需要缩小
        :param adv_range:
        :param rot_range:
        :param model_path: 模型存储路径
        :param log_path: 测试结果保存路径
        :param test_num: 每次测试的递送数量
        """
        super().__init__(target, targets, obs, origin_fig, vessel_skeleton, adv_range, rot_range, stack)

        self.model_path = model_path
        self.log_path = log_path
        self.test_num = test_num

    def run(self):
        threshold = 0.3
        if self.model_path is None:
            self.model_path = []
            for p in ['offlineRL_stack_test\\stack-3\\entropy1.0-policylr1e-5-warm10000-encoder1-PER-full-1\\']:
                # model_path.append(p + '10k\\10k.pt')
                self.model_path.append(p + '30k\\30k.pt')
                self.model_path.append(p + '50k\\50k.pt')
                self.model_path.append(p + '100k\\100k.pt')
                # model_path.append(p + '200k\\200k.pt')
            # model_path = ['offlineRL_stack_test\\stack\\entropy1.0-policylr1e-5-warm10000-encoder1-PER-full-1\\30k\\30k.pt']

        self.log_path = []
        # for p in ['offlineRL_stack_test\\stack-3\\entropy1.0-policylr1e-5-warm10000-encoder1-PER-full-1\\']:
        #     # self.log_path.append(p + '10k')
        #     self.log_path.append(p + '30k')
        #     self.log_path.append(p + '50k')
        #     self.log_path.append(p + '100k')
        # self.log_path.append(p + '200k')
        for p in self.model_path:
            self.log_path.append(os.path.dirname(p))
        # log_path = ['offlineRL_stack_test\\stack\\entropy1.0-policylr1e-5-warm10000-encoder1-PER-full-1\\30k']

        assert len(self.model_path) == len(self.log_path)
        device = torch.device('cpu')
        # encoder = network_architecture.Encoder((140, 140, 2)).to(device)
        aug = ParameterizedReg(aug=LocalSignalMixing(pad=2, fixed_batch=True),
                               parameter_init=0.5, param_grad_fn='alix_param_grad',
                               param_grad_fn_args=[3, 0.535, 1e-20])

        # encoder = encoderExactor((140, 140, 2), 20, 128).to(device)
        if self.dis_map is None:
            self.getDisMap()
        for i in range(len(self.model_path)):
            with open(os.path.join(self.log_path[i], os.pardir, 'args.json'), 'r') as f:
                args = f.read()
                args = json.loads(args)
            if 'encoder' in args.keys():
                encoder = eval(args['encoder'])((140, 140, 4), aug=aug).to(device)
                critic_encoder = eval(args['encoder'])((140, 140, 4), aug=aug).to(device)
            else:
                encoder = ALIXEncoder1((140, 140, 4), aug=aug).to(device)
                critic_encoder = ALIXEncoder1((140, 140, 4), aug=aug).to(device)

            action_dim = 10
            actor_hidden_dim = args['actor_hidden_dim']
            critic_hidden_dim = args['critic_hidden_dim']
            feature_dim = encoder.feature_dim
            actor = nn.Sequential(nn.Linear(feature_dim + action_dim + 1, actor_hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(actor_hidden_dim, actor_hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(actor_hidden_dim, action_dim)).to(device)
            soft_q_nets = nn.Sequential(nn.Linear(feature_dim + action_dim + 1, critic_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(critic_hidden_dim, critic_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(critic_hidden_dim, action_dim)).to(device)

            payload = torch.load(self.model_path[i], map_location='cpu')
            encoder.load_state_dict(payload['encoder'])
            soft_q_nets.load_state_dict(payload['critic'])
            encoder.train(False)
            if 'policy' in payload.keys():
                actor.load_state_dict(payload['policy'])
            elif 'actor' in payload.keys():
                actor.load_state_dict(payload['actor'])
            print(f'successfully load model "{self.model_path[i]}"!')
            done = True
            if not os.path.exists(self.log_path[i]):
                os.makedirs(self.log_path[i])
            test_num = 0
            with open(os.path.join(self.log_path[i], f'{test_num}.txt'), 'w') as f:
                f.write('action;reward;done\n')
            obs = self.reset()
            self.signal.emit(os.path.join(self.log_path[i], f'{test_num}.mp4'), True)
            successful_num = 0
            back_num = 0
            back_sum = 0
            reward_sum = 0
            buffer = []
            while test_num < self.test_num:
                if self.stack:
                    feature = encoder(np.concatenate([obs[0], obs[1]], axis=-1))
                    feature = feature.reshape((feature.shape[0], -1))
                    # last_feature = self.encoder(last_obs)
                    # last_feature = last_feature.reshape((last_feature.shape[0], -1))
                    # feature = self.encoder(obs)
                    # feature = feature.reshape((feature.shape[0], -1))
                    last_action = torch.FloatTensor(obs[2]).unsqueeze(0).to(device)
                    last_action = nn.functional.one_hot(last_action.type(torch.int64).squeeze(-1), num_classes=11)
                    q_input = torch.cat([feature, last_action], dim=1)
                    q = soft_q_nets(q_input)
                    i_1 = actor(q_input)
                    imt = F.log_softmax(i_1, dim=1)
                    imt = imt.exp()
                    imt = (imt / imt.max(1, keepdim=True)[0] > threshold).float()
                    action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True).item()
                else:
                    feature = encoder(obs)
                    action = actor.get_action(feature).item()  # np.ndarray shape = 1
                if action < 5:
                    back_num += 1
                next_obs, reward, done, _ = self.step(action)
                buffer.append(
                    (self.origin_fig, obs, action, reward, next_obs, done, virtualMaster.ActualAdvCurrentOfGw,
                     self.remote))
                reward_sum += reward[1] + reward[2] / 5
                with open(os.path.join(self.log_path[i], f'{test_num}.txt'), 'a') as f:
                    f.write(f'{action};{reward};{done}\n')
                # print(reward, done)
                if done:
                    if reward[0] > 0:
                        successful_num += 1
                        back_sum += back_num
                    self.signal.emit(os.path.join(self.log_path[i], f'{test_num}.mp4'), False)
                    torch.save(buffer, os.path.relpath(os.path.join(self.log_path[i], f'{test_num}.pt')))
                    buffer = []
                    test_num += 1
                    back_num = 0
                    obs = self.reset()
                    done = False
                    # print(test_num, reward)
                    if test_num != self.test_num:
                        with open(os.path.join(self.log_path[i], f'{test_num}.txt'), 'w') as f:
                            f.write('action;reward;done\n')
                        self.signal.emit(os.path.join(self.log_path[i], f'{test_num}.mp4'), True)
                else:
                    obs = next_obs
                # action = np.array([-0.5, -0.7])
            with open(os.path.join(self.log_path[i], os.pardir, os.pardir, 'results.txt'), 'a') as f:
                if successful_num == 0:
                    print(self.model_path[i], f'{successful_num}/{self.test_num}', 0.0, reward_sum / test_num)
                    f.write(f'{self.model_path[i]}, {successful_num}/{self.test_num}, 0.0, {reward_sum / test_num}\n')
                else:
                    print(self.model_path[i], f'{successful_num}/{self.test_num}', back_sum / successful_num,
                          reward_sum / test_num)
                    f.write(
                        f'{self.model_path[i]}, {successful_num}/{self.test_num}, {back_sum / successful_num}, {reward_sum / test_num}\n')


# ------------------------分布式相关函数定义---------------------------------------------#
@retry(stop_max_attempt_number=5, wait_fixed=2000)
def get_remote_class(name: str, namespace: str):
    """
    Try to get ray Remote Class
    :param name:
    :param namespace:
    :return:
    """
    remote_class = ray.get_actor(name=name, namespace=namespace)
    return remote_class


def copy_remote_paras(model, remote_handle):
    """
    copy paras from ray remote class
    :param model: where paras are copied into
    :param remote_handle: ray remote class handle
    :return: the training step in learner
    """
    learner_step, learner_param = ray.get(remote_handle.get_actor_paras.remote())
    for param1, param2 in zip(model.parameters(), learner_param):
        param1.data.copy_(param2.data)
    return learner_step


def copy_remote_paras2(actor, encoder, remote_handle):
    """
    copy paras from ray remote class
    :param model: where paras are copied into
    :param remote_handle: ray remote class handle
    :return: the training step in learner
    """
    learner_step, learner_actor_param, learner_encoder_param = ray.get(
        remote_handle.get_actor_and_encoder_paras.remote())
    for param1, param2 in zip(actor.parameters(), learner_actor_param):
        param1.data.copy_(param2.data)
    for param1, param2 in zip(encoder.parameters(), learner_encoder_param):
        param1.data.copy_(param2.data)
    return learner_step


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)  # 创建一个QApplication，也就是你要开发的软件app
    RLEnv = RLEnvGUI(target=9)  # model1 target 5; model2 target 9
    RLEnv.show()  # 执行QMainWindow的show()方法，显示这个QMainWindow
    sys.exit(app.exec_())
