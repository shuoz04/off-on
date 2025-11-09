import time

from PyQt5.QtCore import QThread, pyqtSignal

import newParas
import socket
import threading



u32_FrameId_Tx = 0  # 发送帧数
u32_FrameId_Rx = 0  # 接收帧数
tickStart = int(time.time() * 1000)
salve_ip = '127.0.0.1'
salve_port = 22
master_port = 1005
socketWatch = socket.socket()  # 定义一个套接字用于监听客户端发来的信息
socketConnect = socket.socket()  # 创建一个负责和客户端通信的套接字
CONST_CLIENT_NUMBER = 1  # 允许的最多客户端的数量
isSlaveConnected = False

# 通讯数据包的封装格式： FrameHead + Data + CheckSum + FrameTail，
# FrameHead为连续的两个0xAA, FrameTail为连续的两个0x55，
# 如果Data中含0xA5、0xAA、0x55（即特殊字符），则在发送该字符之前添加一个控制符0xA5。
# CheckSum为8位校验和，即Data的所有数据之和的低八位
# 如0xAA 0xAA frameID data0 data1 data2 data3 CheckSum 0x55 0x55，
# 如0xAA 0xAA frameID data0 data1 data2 data3 data4 data5 data6 data7 CheckSum 0x55 0x55
NET_FRAME_HEAD = 0xAA  # 帧头
NET_FRAME_TAIL = 0x55  # 帧尾
NET_FRAME_CTRL = 0xA5  # 转义
m_TotalRxLostPackages = 0  # 丢包数
validNetRxBufDataCount = 0  # 接收到的有效字节数

NET_MaxPackageSize = 500  # 一个有效的帧里面最大字节数量
g_RxBuf_Net = bytearray(NET_MaxPackageSize)  # 有效数据的存放数组

initFlag = True  # 初始读数标志符
initGwCount = 0  # 递送前的初始Count
ReceivedSlaveHandshake = False  # 主端是否已经接收到handshake命令
SlaveState = newParas.MS_STAT_E['MS_STAT_RESET']

ActualAdvCountOfGw = int(0)  # 导丝的实际count
ActualAdvSpeedOfGw = int(0)  # 导丝的实际速度
ActualAdvCountOfGw1 = int(0)  # 导丝1的递送count
ActualAdvSpeedOfGw1 = int(0)  # 导丝1的递送速度
ActualRotCountOfGw1 = int(0)  # 导丝1旋转角度count
ActualRotSpeedOfGw1 = int(0)  # 导丝1的旋转速度
GW_ROT_MiddlePos = -155000  # 旋转中间位置（绝对值）
ActualAdvCountOfCath = int(0)  # 导管实际推送长度：单位count
ActualAdvSpeedOfCath = int(0)  # 导管实际推送速度：单位count
CATH_ADV_MiddlePos = 750000  # 推送中间位置（绝对值）
ActualRotCountOfCath = int(0)  # 导管实际旋转角度：单位count
ActualRotSpeedOfCath = int(0)  # 导管实际旋转速度：单位count
g_channel = int(1)  # 当前通道

ActualAdvCurrentOfGw = int(0)
ActualAdvTorqueOfGw = int(0)  # 扭矩传感器的量程是5k到15k，零点是10k

global flag_ballon_install
flag_ballon_install = 0

# 记录同步帧信息及时延
tSyncFrame = {'FrameId': int(-1), 'SendTime': int(-1), 'ReceiveTime': int(-1), 'Delta': int(-1), 'TimeDelay': int(-1)}


def sendTargetVelocity_WithFrameIdAndTime(node: int, targetVelocity: int):
    """
    向从端设置目标运行速度
    :param node: 从端对应的电机
    :param targetVelocity: 电机的期望速度
    :return:
    """
    assert node in newParas.SLAVE_EPOS_NODEID_E.values(), '不存在节点%d对应的电机' % node
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    tmpNetMsg = bytearray(24)

    tmpNetMsg[0:1] = newParas.MS_COM_PROTOCOL_STDID_E['M2S_CMD_VELOCITY'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0  # RTR
    tmpNetMsg[3] = 5  # DLC
    tmpNetMsg[4] = node
    if targetVelocity < 0:
        targetVelocity += 2 ** 32
    tmpNetMsg[5:9] = targetVelocity.to_bytes(4, 'little')
    tmpNetMsg[9:12] = 0x00.to_bytes(2, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')  # 传输方向
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')  # 帧序号
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')  # 发送时刻
    PackAndSendNetMessage(tmpNetMsg)
    # return tmpNetMsg


def sendTargetPosition_WithFrameIdAndTime(node: int, targetPosition: int):
    """
    向从端设置目标运行速度
    :param node: 从端对应的电机
    :param targetPosition: 电机的期望位置
    :return:
    """
    assert node in newParas.SLAVE_EPOS_NODEID_E.values(), '不存在节点%d对应的电机' % node
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    tmpNetMsg = bytearray(24)

    tmpNetMsg[0:1] = newParas.MS_COM_PROTOCOL_STDID_E['M2S_CMD_POSITION'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0  # RTR
    tmpNetMsg[3] = 5  # DLC
    tmpNetMsg[4] = node
    if targetPosition < 0:
        targetPosition += 2 ** 32
    tmpNetMsg[5:9] = targetPosition.to_bytes(4, 'little')
    tmpNetMsg[9:12] = 0x00.to_bytes(2, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')  # 传输方向
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')  # 帧序号
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')  # 发送时刻
    PackAndSendNetMessage(tmpNetMsg)
    # return tmpNetMsg


def sendSyncFrame_WithFrameIdAndTime():
    """
    主端发送同步帧
    :return:
    """
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = newParas.MS_COM_PROTOCOL_STDID_E['M2S_CMD_SyncFrame'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0  # RTR
    tmpNetMsg[3] = 0x04  # DLC
    tmpNetMsg[4] = 0x01
    tmpNetMsg[5:12] = 0x00.to_bytes(7, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')  # 传输方向
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')  # 帧序号
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')  # 发送时刻
    PackAndSendNetMessage(tmpNetMsg)
    tSyncFrame['FrameId'] = int(u32_FrameId_Tx)
    tSyncFrame['SendTime'] = int(g_u32RunTime)


def sendHeartBeatFrame_WithFrameIdAndTime():
    """
    发送心跳帧
    :return:
    """
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = newParas.MS_COM_PROTOCOL_STDID_E['M2S_CMD_HeartBeat'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0  # RTR
    tmpNetMsg[3] = 0x04  # DLC
    tmpNetMsg[4] = 0x01
    tmpNetMsg[5:12] = 0x00.to_bytes(7, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')  # 传输方向
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')  # 帧序号
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')  # 发送时刻
    PackAndSendNetMessage(tmpNetMsg)



def sendTargetModeOfOperation_WithFrameIdAndTime(node: int, targetMode: str):
    """
    向从端设置电机控制模式
    :param node: 从端对应的电机
    :param targetMode: 控制模式
    :return:
    """
    assert node in newParas.SLAVE_EPOS_NODEID_E.values(), '不存在节点%d对应的电机' % node
    assert targetMode in newParas.OPERATION_MOD_E.keys(), '不存在运动模式' + targetMode
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    tmpNetMsg = bytearray(24)

    tmpNetMsg[0:1] = newParas.MS_COM_PROTOCOL_STDID_E['M2S_CMD_OperationMode'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x02  # DLC
    tmpNetMsg[4:5] = node.to_bytes(1, 'little')
    tmpNetMsg[5:6] = newParas.OPERATION_MOD_E[targetMode].to_bytes(1, 'little')
    tmpNetMsg[6:12] = 0x00.to_bytes(6, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')  # 传输方向
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')  # 帧序号
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')  # 发送时刻
    PackAndSendNetMessage(tmpNetMsg)
    # return tmpNetMsg


def sendGWRemove():
    """
    向从端发送拆卸一号导丝的命令
    :return:
    """
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1

    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = newParas.MS_COM_PROTOCOL_STDID_E['M2S_CMD_GW_REMOVE1'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x04  # DLC
    tmpNetMsg[4:8] = 0x00.to_bytes(4, 'little')
    tmpNetMsg[8:12] = 0x00.to_bytes(4, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')  # 传输方向
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')  # 帧序号
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')  # 发送时刻
    PackAndSendNetMessage(tmpNetMsg)


def sendGWInstall():
    """
    向从端发送安装一号导丝的命令
    :return:
    """
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1

    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = newParas.MS_COM_PROTOCOL_STDID_E['M2S_CMD_GW_INSTALL1'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x04  # DLC
    tmpNetMsg[4:8] = 0x01.to_bytes(4, 'little')
    tmpNetMsg[8:12] = 0x00.to_bytes(4, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')  # 传输方向
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')  # 帧序号
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')  # 发送时刻
    PackAndSendNetMessage(tmpNetMsg)

def Control_Change():
    """
    向从端发送模式切换命令 由于速度模式用于球囊 位置模式用于导丝 当用速度模式推导丝后切换位置模式控制到会回到初始点
    :return:
    """
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1

    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = newParas.MS_COM_PROTOCOL_STDID_E['M2S_CONTROL_CHANGE'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x01  # DLC
    tmpNetMsg[4:8] = 0x01.to_bytes(4, 'little')
    tmpNetMsg[8:12] = 0x00.to_bytes(4, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')  # 传输方向
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')  # 帧序号
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')  # 发送时刻
    PackAndSendNetMessage(tmpNetMsg)

def Operate_Send():
    """
    向新从端发送Operate命令
    :return:
    """
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1

    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = newParas.MS_COM_PROTOCOL_STDID_E['Operate_New'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x06  # DLC
    tmpNetMsg[4] = 0x11
    tmpNetMsg[5] = 0x22
    tmpNetMsg[6] = 0x33
    tmpNetMsg[7] = 0x44
    tmpNetMsg[8] = 0x55
    tmpNetMsg[9] = 0x66
    tmpNetMsg[10] = 0x00
    tmpNetMsg[11] = 0x00
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')  # 传输方向
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')  # 帧序号
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')  # 发送时刻
    PackAndSendNetMessage(tmpNetMsg)

def Voice_Play(num):
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    voice_list = [0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xDF]
    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = newParas.MS_COM_PROTOCOL_STDID_E['Voice_Play'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x01  # DLC
    tmpNetMsg[4] = voice_list[num]
    tmpNetMsg[5] = 0x00
    tmpNetMsg[6] = 0x00
    tmpNetMsg[7] = 0x00
    tmpNetMsg[8] = 0x00
    tmpNetMsg[9] = 0x00
    tmpNetMsg[10] = 0x00
    tmpNetMsg[11] = 0x00
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')  # 传输方向
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')  # 帧序号
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')  # 发送时刻
    PackAndSendNetMessage(tmpNetMsg)

def Install_Ballon():
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = newParas.MS_COM_PROTOCOL_STDID_E['Install_Ballon'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x01  # DLC
    tmpNetMsg[4] = 0x31
    tmpNetMsg[5] = 0x00
    tmpNetMsg[6] = 0x00
    tmpNetMsg[7] = 0x00
    tmpNetMsg[8] = 0x00
    tmpNetMsg[9] = 0x00
    tmpNetMsg[10] = 0x00
    tmpNetMsg[11] = 0x00
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')  # 传输方向
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')  # 帧序号
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')  # 发送时刻
    PackAndSendNetMessage(tmpNetMsg)


def PackAndSendNetMessage(byteArray: bytearray):
    """
    将数组打包（添加帧头、帧尾和校验和）并发送
    :param byteArray:发送数组（无帧头、帧尾和校验和）
    :return:
    """
    # assert isSlaveConnected == True, "从端未连接！"

    tmpByteArray = bytearray(100)
    CheckSum = 0
    # len = 0
    tmpByteArray[0] = NET_FRAME_HEAD
    tmpByteArray[1] = NET_FRAME_HEAD
    length = 2  # 因为与关键词重名，因此这里不命名为c#程序中的len
    for iter in range(len(byteArray)):
        if byteArray[iter] == NET_FRAME_CTRL or byteArray[iter] == NET_FRAME_HEAD or byteArray[iter] == NET_FRAME_TAIL:
            tmpByteArray[length] = NET_FRAME_CTRL
            length += 1
        tmpByteArray[length] = byteArray[iter]
        CheckSum += byteArray[iter]
        CheckSum = CheckSum % 256
        length += 1
    if CheckSum == NET_FRAME_CTRL or CheckSum == NET_FRAME_HEAD or CheckSum == NET_FRAME_TAIL:
        tmpByteArray[length] = NET_FRAME_CTRL
        length += 1
    tmpByteArray[length] = CheckSum
    length += 1
    tmpByteArray[length] = NET_FRAME_TAIL
    length += 1
    tmpByteArray[length] = NET_FRAME_TAIL
    length += 1
    socketConnect.send(tmpByteArray)

def getLocalIP():
    """
    查询本机ip地址，即c#程序中的btnGetLocalIP_Click函数
    :return:
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def ServerStart(ip, port):
    """
    Tcp服务器建立，即c#程序中的btnServerConn_Click函数
    :return:
    """
    global socketWatch
    socketWatch.bind(("", port))
    socketWatch.listen(CONST_CLIENT_NUMBER)

    threadWatch = threading.Thread(target=ThreadFunc_SendMsgToSlave)
    print(f'主端服务器启动 主端ip为{ip} 端口为{port}')
    # python的进程分前后台吗？在c#程序中进程分前后台
    threadWatch.start()

def ClientStart(ip, port):
    threadWatch = threading.Thread(target=ThreadFunc_SendMsgToServer, args=(ip, port,))
    print(f'连接服务端 ip为{ip} 端口为{port}')
    threadWatch.start()

def ThreadFunc_SendMsgToSlave():
    """
    发送握手、同步、心跳帧至从端
    :return:
    """
    u16_SyncIndex = 0
    u8_HeartBeatIndex = 0x00
    u16_PrintIndex = 950
    global socketConnect, isSlaveConnected
    while not isSlaveConnected:
        socketConnect, _ = socketWatch.accept()  # 这个方法会阻断当前的线程：持续不断监听
        print('从端连接成功')
        thr = threading.Thread(target=ThreadFunc_DataReceive, args=(socketConnect,))
        thr.start()
        isSlaveConnected = True

    while SlaveState is not newParas.MS_STAT_E['MS_STAT_OK']:
        time.sleep(0.1)  # 注意在python里time.sleep()只对当前进程有效，因此等价于c#中的Thread.Sleep()
    print('开始正常通信')
    time.sleep(2)
    # threadContorl = threading.Thread(target=ThreadFunc_SendTargetVelocity)
    # threadContorl.start()
    # sendTargetVelocity_WithFrameIdAndTime(newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 10)
    while True:
        u16_SyncIndex += 1
        if u16_SyncIndex == 1000:
            u16_SyncIndex = 0
            sendSyncFrame_WithFrameIdAndTime()
        u8_HeartBeatIndex += 1
        if u8_HeartBeatIndex == 1:
            u8_HeartBeatIndex = 0
            if u32_FrameId_Tx < 1000000:
                sendHeartBeatFrame_WithFrameIdAndTime()

def ThreadFunc_SendMsgToServer(ip, port):
    """
    发送握手、同步、心跳帧至从端
    :return:
    """
    u16_SyncIndex = 0
    u8_HeartBeatIndex = 0x00
    u16_PrintIndex = 950

    global socketWatch, SlaveState
    # 创建套接字
    socketWatch = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 设置端口复用，使程序退出后端口马上释放
    socketWatch.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

    # 客户端连接程序
    server_addr = (ip, int(port))  # IP地址 + 端口

    global socketConnect, isSlaveConnected
    while not isSlaveConnected:
        socketWatch.connect(server_addr)  # 连接
        socketConnect = socketWatch
        print('从端连接成功(client模式)')
        thr = threading.Thread(target=ThreadFunc_DataReceive, args=(socketConnect,))
        thr.start()
        print("THR OK")
        isSlaveConnected = True
        Operate_Send()

    SlaveState = newParas.MS_STAT_E['MS_STAT_OK']
    while SlaveState is not newParas.MS_STAT_E['MS_STAT_OK']:
        time.sleep(0.1)  # 注意在python里time.sleep()只对当前进程有效，因此等价于c#中的Thread.Sleep()
    print('开始正常通信')
    time.sleep(2)
    # threadContorl = threading.Thread(target=ThreadFunc_SendTargetVelocity)
    # threadContorl.start()
    # sendTargetVelocity_WithFrameIdAndTime(newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 10)
    while True:
        time.sleep(1)
        u16_SyncIndex += 1
        if u16_SyncIndex == 1000:
            u16_SyncIndex = 0
            sendSyncFrame_WithFrameIdAndTime()
        u8_HeartBeatIndex += 1
        if u8_HeartBeatIndex == 1:
            u8_HeartBeatIndex = 0
            if u32_FrameId_Tx < 1000000:
                sendHeartBeatFrame_WithFrameIdAndTime()


# def ThreadFunc_DataReceive(socketServer):
#     global u32_FrameId_Rx, ActualAdvCountOfGw, ActualAdvSpeedOfGw, ActualAdvCurrentOfGw, ActualAdvTorqueOfGw
#     # isRxValidPackage = False
#     canMessage = {}
#     time.sleep(0.5)
#     while True:
#         byteArray = socketServer.recv(100)
#         for index in range(len(byteArray)):
#             tmpByte = byteArray[index]
#             isRxValidPackage = ParseRxByteFromNet(tmpByte)
#             if isRxValidPackage:
#                 u32_FrameId_Rx += 1
#                 canMessage['StdId'] = int.from_bytes(g_RxBuf_Net[8:10], 'little')
#                 # if canMessage['StdId'] == newParas.MS_COM_PROTOCOL_STDID_E['S2M_RPL_SyncFrame']:  # 主端计算网络延时
#                 if canMessage['StdId'] == newParas.MS_COM_PROTOCOL_STDID_E['S2M_RPL_PosVelTorqAndSensor']:
#                     ActualAdvCountOfGw = int.from_bytes(g_RxBuf_Net[10:14], 'little', signed=True)
#                     ActualAdvSpeedOfGw = int.from_bytes(g_RxBuf_Net[14:18], 'little', signed=True)
#                     ActualAdvCurrentOfGw = int.from_bytes(g_RxBuf_Net[18:20], 'little', signed=True)
#                     ActualAdvTorqueOfGw = int.from_bytes(g_RxBuf_Net[20:22], 'little')
#                     # print(ActualAdvCountOfGw, ActualAdvSpeedOfGw, ActualAdvCurrentOfGw, ActualAdvTorqueOfGw)
#                     global initFlag, initGwCount
#                     if initFlag:
#                         initFlag = False
#                         initGwCount = ActualAdvCountOfGw
#                 else:
#                     canMessage['RTR'] = g_RxBuf_Net[10]
#                     canMessage['DLC'] = g_RxBuf_Net[11]
#                     canMessage['data'] = g_RxBuf_Net[12:12 + canMessage['DLC']]
#                     ParseSlaveMsg(canMessage)

def ThreadFunc_DataReceive(socketServer):
    global u32_FrameId_Rx, ActualAdvCountOfGw, ActualAdvSpeedOfGw, ActualAdvCurrentOfGw, ActualAdvTorqueOfGw
    # isRxValidPackage = False
    canMessage = {}
    time.sleep(0.5)
    while True:
        byteArray = socketServer.recv(100)
        for index in range(len(byteArray)):
            tmpByte = byteArray[index]
            isRxValidPackage = ParseRxByteFromNet(tmpByte)
            if isRxValidPackage:
                u32_FrameId_Rx += 1
                canMessage['StdId'] = int.from_bytes(g_RxBuf_Net[0:1], 'little')
                # if canMessage['StdId'] == newParas.MS_COM_PROTOCOL_STDID_E['S2M_RPL_SyncFrame']:  # 主端计算网络延时
                if canMessage['StdId'] == newParas.MS_COM_PROTOCOL_STDID_E['S2M_RPL_PosVelTorqAndSensor']:
                    ActualAdvCountOfGw = int.from_bytes(g_RxBuf_Net[4:8], 'little', signed=True)
                    ActualAdvSpeedOfGw = int.from_bytes(g_RxBuf_Net[8:12], 'little', signed=True)
                    # ActualAdvCurrentOfGw = int.from_bytes(g_RxBuf_Net[12:16], 'little', signed=True)
                    # ActualAdvTorqueOfGw = int.from_bytes(g_RxBuf_Net[16:20], 'little')

                    # 【电流】已安装器械或非安装器械
                    if g_RxBuf_Net[14] != 0 or g_RxBuf_Net[14] == 0:
                        # 将两个16进制数拼接
                        hex1 = g_RxBuf_Net[21]  # 第一个十六进制值 21
                        hex2 = g_RxBuf_Net[20]  # 第二个十六进制值 20
                        combined_hex = (hex1 << 8) | hex2
                        # print(hex(combined_hex))

                        # 转有符号十进制扭矩数据
                        ActualAdvCurrentOfGw = combined_hex if combined_hex < 0x8000 else combined_hex - 0x10000
                        # print(ActualAdvCurrentOfGw)

                    # 【扭矩】已安装器械或非安装器械
                    if g_RxBuf_Net[14] != 0 or g_RxBuf_Net[14] == 0:
                        # 将两个16进制数拼接
                        hex1 = g_RxBuf_Net[23]  # 第一个十六进制值 21
                        hex2 = g_RxBuf_Net[22]  # 第二个十六进制值 20
                        combined_hex = (hex1 << 8) | hex2
                        # print(hex(combined_hex))

                        # 转有符号十进制扭矩数据
                        ActualAdvTorqueOfGw = combined_hex if combined_hex < 0x8000 else combined_hex - 0x10000
                        # print(ActualAdvTorqueOfGw)

                    # print(ActualAdvCountOfGw, ActualAdvSpeedOfGw, ActualAdvCurrentOfGw, ActualAdvTorqueOfGw)
                    global initFlag, initGwCount
                    if initFlag:
                        initFlag = False
                        initGwCount = ActualAdvCountOfGw

                elif g_RxBuf_Net[0] == 0x32 and g_RxBuf_Net[3] == 0x05 and g_RxBuf_Net[7] == 0x02:  # 球囊完成
                    global flag_ballon_install
                    flag_ballon_install = 1
                    print("球囊安装完成")

                else:
                    canMessage['RTR'] = g_RxBuf_Net[2]
                    canMessage['DLC'] = g_RxBuf_Net[3]
                    canMessage['data'] = g_RxBuf_Net[4:12]
                    ParseSlaveMsg(canMessage)


# 以下变量用于解析：这些变量只能用于ParseRxByteFromNet函数！！！
NET_LastByte = 0
NET_BeginFlag = False
NET_CtrlFlag = False
NET_RevOffset = 0  # 数据的字节数
NET_CheckSum = 0  # 校验和


def ParseRxByteFromNet(data: int) -> bool:
    """
    检测收到的1个字节网络数据是否是网络帧的结尾
    :param data: 待检测字节
    :return: TRUE代表成功解析1帧数据，有效数据保存在g_RxBuf_Net数组中，长度为g_ValidDataCount_Net
    """
    global NET_RevOffset, NET_BeginFlag, NET_LastByte, m_TotalRxLostPackages, validNetRxBufDataCount, NET_CheckSum, g_RxBuf_Net, NET_CtrlFlag
    if (data == NET_FRAME_HEAD and NET_LastByte == NET_FRAME_HEAD) or NET_RevOffset > NET_MaxPackageSize:
        if 25 > NET_RevOffset > 0:
            m_TotalRxLostPackages += 1
        # reset
        NET_RevOffset = 0
        NET_BeginFlag = True
        NET_LastByte = data
        return False
    if data == NET_FRAME_TAIL and NET_LastByte == NET_FRAME_TAIL and NET_BeginFlag:
        NET_RevOffset -= 1  # 得到除去头尾和控制符的数据字节数
        validNetRxBufDataCount = NET_RevOffset - 1  # 得到除去头尾、控制符和校验和的数据字节数
        NET_CheckSum -= NET_FRAME_TAIL
        NET_CheckSum -= g_RxBuf_Net[validNetRxBufDataCount]
        NET_LastByte = data
        NET_BeginFlag = False
        NET_CheckSum = NET_CheckSum % 256
        if NET_CheckSum == g_RxBuf_Net[validNetRxBufDataCount]:
            NET_CheckSum = 0
            return True
        else:
            m_TotalRxLostPackages += 1
            NET_CheckSum = 0
            return False
    NET_LastByte = data
    if NET_BeginFlag:
        if NET_CtrlFlag:
            g_RxBuf_Net[NET_RevOffset] = data
            NET_RevOffset += 1
            NET_CheckSum += data
            NET_CtrlFlag = False
            NET_LastByte = NET_FRAME_CTRL  # 为什么这里是NET_FRAME_CTRL而不是data
        elif data == NET_FRAME_CTRL:
            NET_CtrlFlag = True
        else:
            g_RxBuf_Net[NET_RevOffset] = data
            NET_RevOffset += 1
            NET_CheckSum += data
    return False


def ParseSlaveMsg(canMessage: dict):
    """
    解析接收到的从端信息
    :param canMessage: 从端信息
    :return:
    """
    global ReceivedSlaveHandshake, u32_FrameId_Tx, SlaveState, ActualAdvCountOfGw1, ActualAdvSpeedOfGw1, \
        ActualRotCountOfGw1, ActualRotSpeedOfGw1, ActualAdvCountOfCath, ActualAdvSpeedOfCath, ActualRotCountOfCath, \
        ActualRotSpeedOfCath, g_channel
    if canMessage['StdId'] == newParas.MS_COM_PROTOCOL_STDID_E['S2M_RPL_HANDSHAKE']:
        if canMessage['DLC'] == 1:
            # 如果从端从未发过handshake信号且从端从未发过operate信号且收到的信息为handshake信息
            if ReceivedSlaveHandshake != True and SlaveState != newParas.MS_STAT_E['MS_STAT_OK'] and canMessage['data'][0] \
                    == newParas.MS_STAT_E['MS_STAT_OK']:
                print('主端收到handshake')
                ReceivedSlaveHandshake = True

                # 主端发送
                u32_FrameId_Tx += 1
                tmpNetMsg = bytearray(13)
                tmpNetMsg[0:4] = u32_FrameId_Tx.to_bytes(4, 'little')  # 帧序号
                g_u32RunTime = int(time.time() * 1000) - tickStart
                tmpNetMsg[4:8] = g_u32RunTime.to_bytes(4, 'little')  # 发送时刻
                tmpNetMsg[8:10] = newParas.MS_COM_PROTOCOL_STDID_E['M2S_CMD_OPERATION'].to_bytes(2, 'little')
                tmpNetMsg[10] = 0x00  # RTRs
                tmpNetMsg[11] = 0x01  # DLC
                tmpNetMsg[12] = 0x01
                PackAndSendNetMessage(tmpNetMsg)
                print('主端发送operate')
        return
    elif canMessage['StdId'] == newParas.MS_COM_PROTOCOL_STDID_E['S2M_RPL_OPERATION']:
        if ReceivedSlaveHandshake and SlaveState != newParas.MS_STAT_E['MS_STAT_OK'] and canMessage['DLC'] == 1 and \
                canMessage['data'][0] == 1:
            SlaveState = newParas.MS_STAT_E['MS_STAT_OK']
            print('从端回复operate')
        return
    elif canMessage['StdId'] == newParas.MS_COM_PROTOCOL_STDID_E['S2M_RPL_PosAndVel']:
        if canMessage['DLC'] == 8:
            if canMessage['RTR'] == newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV']:
                ActualAdvCountOfGw1 = int.from_bytes(canMessage['data'][0:4], 'little')
                ActualAdvSpeedOfGw1 = int.from_bytes(canMessage['data'][4:8], 'little')
                return
            elif canMessage['RTR'] == newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT']:
                ActualRotCountOfGw1 = int.from_bytes(canMessage['data'][0:4], 'little')
                ActualRotSpeedOfGw1 = int.from_bytes(canMessage['data'][4:8], 'little')
                ActualRotCountOfGw1 = ActualRotCountOfGw1 - GW_ROT_MiddlePos
                return
            elif canMessage['RTR'] == newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_CATH_ADV']:
                ActualAdvCountOfCath = int.from_bytes(canMessage['data'][0:4], 'little')
                ActualAdvSpeedOfCath = int.from_bytes(canMessage['data'][4:8], 'little')
                ActualAdvCountOfCath = ActualAdvCountOfCath - CATH_ADV_MiddlePos
                return
            elif canMessage['RTR'] == newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_CATH_ROT']:
                ActualRotCountOfCath = int.from_bytes(canMessage['data'][0:4], 'little')
                ActualRotSpeedOfCath = int.from_bytes(canMessage['data'][4:8], 'little')
                return
        return
    elif canMessage['StdId'] == newParas.MS_COM_PROTOCOL_STDID_E['S2M_RPL_CHANNEL']:
        if canMessage['DLC'] == 1:
            g_channel = canMessage['data'][0]
        return

def ThreadFunc_SendTargetVelocity(v):
    sendTargetVelocity_WithFrameIdAndTime(newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], v)


class masterThread(QThread):
    signal = pyqtSignal(bool)

    def __init__(self):
        """
        环境初始化
        """
        # robot initialize
        super().__init__()

    def run(self):
        print('正在初始化虚拟主端')
        ClientStart(ip='192.168.0.22', port=22)
        while SlaveState is not newParas.MS_STAT_E['MS_STAT_OK']:
            time.sleep(0.1)
        time.sleep(5)
        sendGWInstall()
        time.sleep(0.5)
        sendTargetModeOfOperation_WithFrameIdAndTime(newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'],
                                                     'PVM')
        time.sleep(0.5)
        sendTargetModeOfOperation_WithFrameIdAndTime(newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'],
                                                     'PVM')
        self.signal.emit(True)

if __name__ == '__main__':
    ClientStart(ip='192.168.0.22', port=22)
    while SlaveState is not newParas.MS_STAT_E['MS_STAT_OK']:
        time.sleep(0.1)
    time.sleep(5)
    sendGWInstall()
    time.sleep(0.5)
    sendTargetModeOfOperation_WithFrameIdAndTime(newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'],
                                                               'PVM')
    time.sleep(0.5)
    sendTargetModeOfOperation_WithFrameIdAndTime(newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'],
                                                               'PVM')
    print('start')

    action_set = [(-5, -10), (-5, -5), (-5, 0), (-5, 5), (-5, 10),
                  (5, -10), (5, -5), (5, 0), (5, 5), (5, 10)]
    import keyboard
    while True:
        if keyboard.is_pressed('w'):
            action = 7
        elif keyboard.is_pressed('s'):
            action = 2
        else:
            action = None
        if action is not None:
            if keyboard.is_pressed('h'):
                action -= 2
            elif keyboard.is_pressed('j'):
                action -= 1
            elif keyboard.is_pressed('k'):
                action += 1
            elif keyboard.is_pressed('l'):
                action += 2

            action = int(action)
            print(action)
            adv = action_set[action][0] * 2
            rot = action_set[action][1] * 5
            sendTargetVelocity_WithFrameIdAndTime(newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], adv)
            time.sleep(0.05)
            sendTargetVelocity_WithFrameIdAndTime(newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'], rot)
            time.sleep(0.5)
            sendTargetVelocity_WithFrameIdAndTime(newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 0)
            time.sleep(0.05)
            sendTargetVelocity_WithFrameIdAndTime(newParas.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'], 0)
            time.sleep(0.05)
        if keyboard.is_pressed('p'):
            sendGWInstall()
        elif keyboard.is_pressed('o'):
            sendGWRemove()
