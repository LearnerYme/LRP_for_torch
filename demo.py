import numpy as np
import torch
import torchvision.transforms as tr
import matplotlib.pyplot as plt
import matplotlib.image as img
from util.model import Net
from util.lrp import lrp

#load model
lenet = Net()
lenet.load_state_dict(torch.load('./lenet.pth'))
lenet.eval()#if your model contains batch normalization or drop out

ipt = plt.imread('2.png').reshape(1,1,28,28)
ipt = torch.from_numpy(ipt).float()

lrpt = lrp(1e-6)
lrpt.Rinit(lenet,ipt,False)#if your model contains softmax in the last layer, set False
lrpt.set_('fc',lenet.fc2_value,lenet.fc3.weight)
lrpt.fc()
lrpt.set_('fc',lenet.fc1_value,lenet.fc2.weight)
lrpt.fc()
lrpt.set_('fc',lenet.flatten_value,lenet.fc1.weight)
lrpt.fc()
lrpt.set_('flatten',lenet.pool2_value)
lrpt.flatten()
lrpt.set_('avepool',lenet.conv2_value,lenet.pool2_value,2,2,2,2)
lrpt.avepool()
lrpt.set_('conv',lenet.pool1_value,lenet.conv2.weight,1,1)
lrpt.conv()
lrpt.set_('avepool',lenet.conv1_value,lenet.pool1_value,2,2,2,2)
lrpt.avepool()
lrpt.set_('conv',ipt,lenet.conv1.weight,1,1)
lrpt.conv()

R_ipt = lrpt.R.numpy().reshape(28,28)
img.imsave('lrp_2.png',R_ipt)