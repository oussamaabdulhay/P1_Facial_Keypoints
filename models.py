
## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        

        self.conv1 = nn.Conv2d(1, 32, 3)#111 after 1 conv and pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3) #54 after 2 conv and pooling layers
        self.conv3 = nn.Conv2d(64, 128, 3)#26 after 3 conv and pooling layers
        self.conv4 = nn.Conv2d(128, 256, 3)#12 after 4 conv and pooling layers
        self.conv5 = nn.Conv2d(256, 512, 3)#5 after 5 conv and pooling layers
        
        
        self.fc1 = nn.Linear(512*5*5, 1280) 
        
        
        self.fc2 = nn.Linear(1280, 640)  
        
        self.fc1_drop = nn.Dropout(p=0.4)

        self.fc3 = nn.Linear(640, 136)
 



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x