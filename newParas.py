# 从端各EPOS的node id
SLAVE_EPOS_NODEID_E = {'SLAVE_NODEID_GW_ADV': 0x01,  # 导丝推送
                       'SLAVE_NODEID_GW_ROT': 0x02,  # 导丝旋转
                       'SLAVE_NODEID_CATH_ADV': 0x03,  # 导管推送
                       'SLAVE_NODEID_CATH_ROT': 0x04,  # 导管旋转
                       'SLAVE_NODEID_YVAL_ROT': 0x05,  # Y阀门开合
                       'SLAVE_NODEID_CH1_FRONT': 0x06,  # 通道1前端夹紧电机
                       'SLAVE_NODEID_CH1_REAR': 0x07,  # 通道1尾部夹紧电机
                       'SLAVE_NODEID_CH2_FRONT': 0x08,  # 通道2前端夹紧电机
                       'SLAVE_NODEID_CH2_REAR': 0x09,  # 通道2尾部夹紧电机
                       }

# 主从通讯协议的StdID
MS_COM_PROTOCOL_STDID_E = {'M2S_CMD_HeartBeat': 0x03,  # 主端发给从端“心跳”，格式：0x11, 00, 00, 01, 00
                           'S2M_RPL_HeartBeat': 0x13,  # 主端发给从端“心跳”，格式：0x12, 00, 00, 01, 00

                           'M2S_CMD_EMER': 0x01,  # 主端发送给从端 “急停”，格式：0x21, 00, 00, 01, 00
                           'S2M_CMD_EMER': 0x11,  # 主端发送给从端 “急停”

                           'M2S_CMD_HANDSHAKE': 0x31,  # 主端发给从端的握手信号，格式：0x31, 00, 00, 01, MS_STAT_OK
                           'S2M_RPL_HANDSHAKE': 0x32,  # 从端发给主端的握手信号

                           'M2S_CMD_OPERATION': 0x2B,  # 主端发给从端：正式开始？格式：0x41, 00, 00, 01, 01
                           'S2M_RPL_OPERATION': 0x3B,  # 从端发给主端：正式开始！格式：0x42, 00, 00, 01, 01

                           'M2S_CMD_CLEARERROR': 0x0F,  # 主端发给从端：清除EPOS的error code
                           'S2M_RPL_ERRORCODE': 0x1F,  # 从端发给主端：EPOS的error code

                           'M2S_CMD_POSITION': 0x07,  # 主端发给从端：各EPOS的期望位置 data[0]=nodeID, data[1-4]
                           'M2S_CMD_VELOCITY': 0x09,  # 主端发给从端：各EPOS的期望速度 data[0]=nodeID, data[1-4]
                           # 注意，这个与主从CAN通信时不一样！！！
                           'S2M_RPL_PosAndVel': 0x37,  # 从端发给主端：各EPOS的实际位置 RTR=nodeID, 位置data[0-3], 速度data[4-7]

                           'M2S_CMD_GW_REMOVE1': 0x21,  # 主端发给从端：松开1号导丝
                           'M2S_CMD_GW_INSTALL1': 0x21,  # 主端发给从端：安装1号导丝

                           'M2S_CMD_CLAMP': 0x21,  # 主端发给从端：各夹紧电机的期望状态
                           'S2M_RPL_CLAMP': 0x31,  # 从端发给主端：各夹紧电机的实际状态

                           'M2S_CMD_OperationMode': 0x05,
                           # 主端发送给从端：EPOS的运动模式，data[0]': nodeid, data[1]=PPM, PVM (OPERATION_MOD_E)

                           'S2M_RPL_TORQUE': 0x1E,  # 从端反馈至主端扭矩值（频率15K-25K Hz）：u16

                           'M2S_CMD_SyncFrame': 0x04,  # 同步帧：格式  主端发送帧号  主端发送时间t1  0xB1, 00, 00, 01, 01
                           'S2M_RPL_SyncFrame': 0x14,
                           # 同步帧：格式  从端发送帧号  从端从接收到发送耗时delta 0xB2, 00, 00, 08, 主端发送帧号 主端发送时间t1

                           'S2M_RPL_CHANNEL': 0x38,  # 从端回复当前通道

                           'S2M_RPL_PosVelTorqAndSensor': 0x17,
                           # 从端发给主端，当前导丝/球囊推送电机的：位置、速度、电机扭矩、传感器频率，格式：0xD2, 0x00, int32位置, int32速度, int16电机扭矩, u16传感器频率

                           'M2S_CMD_PROVELOCITY': 0xE1,  # 主端设置从端电机最大运行速度

                           'M2S_CONTROL_CHANGE': 0x41,  # 速度-位置模式切换

                           'Operate_New': 0x26,  # 新从端Operate

                            'Voice_Play': 0x29,  # 新从端播放语音

                            'Install_Ballon': 0x22

                           }

MS_STAT_E = {'MS_STAT_RESET': 0,  # 复位
             'MS_STAT_INI': 1,  # 正在初始化：自检、检查EPOS
             'MS_STAT_OK': 2,  # ok
             'MS_STAT_ERR': 3  # 错误
             }

# 运动模式
OPERATION_MOD_E = {'MODE_NC': 0,
                   'PPM': 1,  # Profile Position Mode
                   'PVM': 3,  # Profile Velocity Mode
                   'CSP': 8,  # cyclic synchronous position mode
                   'CSV': 9,  # cyclic synchronous velocity mode
                   'CST': 10,  # cyclic synchronous torque mode
                   'HMM': 6,  # Homing Mode
                   }
