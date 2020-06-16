import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #a LeNet model
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,1)
        self.pool = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(400,100)
        self.fc2 = nn.Linear(100,84)
        self.fc3 = nn.Linear(84,10)
        return

    def forward(self,x):
        x = F.relu(self.conv1(x))
        self.conv1_value = x
        x = self.pool(x)
        self.pool1_value = x
        x = F.relu(self.conv2(x))
        self.conv2_value = x
        x = self.pool(x)
        self.pool2_value = x
        x = x.view(-1,400)
        self.flatten_value = x
        x = F.relu(self.fc1(x))
        self.fc1_value = x
        x = F.relu(self.fc2(x))
        self.fc2_value = x
        x = F.softmax(self.fc3(x),1)
        return x

