from PyQt5.QtCore import QThread, pyqtSignal
import paras
import virtualMaster
import time


class masterThread(QThread):
    signal = pyqtSignal(bool)

    def __init__(self):
        """
        环境初始化
        """
        # robot initialize
        super().__init__()

    def run(self):
        virtualMaster.printMessage('正在初始化虚拟主端')
        virtualMaster.ServerStart()
        while virtualMaster.SlaveState is not paras.MS_STAT_E['MS_STAT_OK']:
            time.sleep(0.1)
        time.sleep(1)
        virtualMaster.sendGWInstall()
        time.sleep(0.5)
        virtualMaster.sendTargetModeOfOperation_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'],
                                                                   'PVM')
        time.sleep(0.5)
        virtualMaster.sendTargetModeOfOperation_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT'],
                                                                   'PVM')
        virtualMaster.printMessage('虚拟主端初始化完成')
        self.signal.emit(True)
