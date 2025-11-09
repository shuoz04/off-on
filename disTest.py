import matplotlib.pyplot as plt
import torch

dis = torch.load('D:\LH\程序\自主递送\VirtualMaster2\camera_log/dis.pt')
plt.imshow(dis)
plt.show()