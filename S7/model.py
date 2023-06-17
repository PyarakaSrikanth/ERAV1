import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2) # 28>26 | 1>3 | 1>1
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #26>24>12 | 3>5>6 | 1>1>2
        x = F.relu(self.conv3(x), 2) # 12>10 | 6>10 | 2>2
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # 10>8>4 | 10>14>16 | 2>2>4
        x = x.view(-1, 4096) # 4*4*256 = 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

 ## Model Version 1 
 #Model_1
 #*******************************************************************#
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=0,bias=False) 
        self.bn2 = nn.BatchNorm2d(num_features=64) 
        self.tns1 = nn.Conv2d(64,32,1,padding=0,bias=False) 
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.conv3 = nn.Conv2d(32,64,3, padding=0,bias=False) 
        self.bn3 = nn.BatchNorm2d(num_features=64) 
        self.conv4 = nn.Conv2d(64,64,3, padding=0,bias=False) 
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = nn.Conv2d(64,64,3, padding=0,bias=False) 
        self.bn5 = nn.BatchNorm2d(num_features=64)    
        
        self.gpool = nn.AvgPool2d(6)

        self.conv6 = nn.Conv2d(64,10,1,padding=0,bias=False)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(self.tns1(x))

      

        x = self.bn3(F.relu(self.conv3(x)))       
        x = self.bn4(F.relu(self.conv4(x)))


        x = self.bn5(F.relu(self.conv5(x)))

        x = self.gpool(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x,dim=-1)    
 #*******************************************************************#  

 #Model version 2
 # Objective is to reduce Parameters and train with decent test accuracy
 #********************************************************************#
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 9, 3, padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=9)
        self.conv2 = nn.Conv2d(9, 18, 3, padding=0,bias=False) 
        self.bn2 = nn.BatchNorm2d(num_features=18) 
        self.tns1 = nn.Conv2d(18,12,1,padding=0,bias=False) 
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.conv3 = nn.Conv2d(12,18,3, padding=0,bias=False) 
        self.bn3 = nn.BatchNorm2d(num_features=18) 
        self.conv4 = nn.Conv2d(18,18,3, padding=0,bias=False) 
        self.bn4 = nn.BatchNorm2d(num_features=18)
        self.conv5 = nn.Conv2d(18,18,3, padding=0,bias=False) 
        self.bn5 = nn.BatchNorm2d(num_features=18)    
        
        self.gpool = nn.AvgPool2d(5)

        self.conv6 = nn.Conv2d(18,10,1,padding=0,bias=False)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(self.tns1(x))

      

        x = self.bn3(F.relu(self.conv3(x)))       
        x = self.bn4(F.relu(self.conv4(x)))


        x = self.bn5(F.relu(self.conv5(x)))

        x = self.gpool(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x,dim=-1)      

#**********************************************************************#   
 

 #Model version 3
 # Objective is to handle overfitting and improve accuracy applied transformation
 #apply rotation
 #********************************************************************#
class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 9, 3, padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=9)
        self.conv2 = nn.Conv2d(9, 18, 3, padding=0,bias=False) 
        self.bn2 = nn.BatchNorm2d(num_features=18) 
        self.tns1 = nn.Conv2d(18,12,1,padding=0,bias=False) 
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.conv3 = nn.Conv2d(12,18,3, padding=0,bias=False) 
        self.bn3 = nn.BatchNorm2d(num_features=18) 
        self.conv4 = nn.Conv2d(18,18,3, padding=0,bias=False) 
        self.bn4 = nn.BatchNorm2d(num_features=18)
        self.conv5 = nn.Conv2d(18,18,3, padding=0,bias=False) 
        self.bn5 = nn.BatchNorm2d(num_features=18)    
        self.drop = nn.Dropout2d(0.01)
        self.gpool = nn.AvgPool2d(5)

        self.conv6 = nn.Conv2d(18,10,1,padding=0,bias=False)

    def forward(self, x):
        x = self.drop(self.bn1(F.relu(self.conv1(x))))
        x = self.drop(self.bn2(F.relu(self.conv2(x))))
        x = self.pool1(self.tns1(x))

      

        x = self.drop(self.bn3(F.relu(self.conv3(x))))       
        x = self.drop(self.bn4(F.relu(self.conv4(x))))


        x = self.drop(self.bn5(F.relu(self.conv5(x))))

        x = self.gpool(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x,dim=-1)
#**********************************************************************#   
 

 #Model version 4
 # Target of 99.4 accuracy consistantly and parameters < 8000
 # Objective is to handle overfitting and improve accuracy applied transformation
 #apply rotation
 #********************************************************************#

class Model_4(nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, 3, padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=7)
        self.conv2 = nn.Conv2d(7, 16, 3, padding=0,bias=False) 
        self.bn2 = nn.BatchNorm2d(num_features=16) 
        self.tns1 = nn.Conv2d(16,12,1,padding=0,bias=False) 
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.conv3 = nn.Conv2d(12,16,3, padding=0,bias=False) 
        self.bn3 = nn.BatchNorm2d(num_features=16) 
        self.conv4 = nn.Conv2d(16,16,3, padding=0,bias=False) 
        self.bn4 = nn.BatchNorm2d(num_features=16)
        self.conv5 = nn.Conv2d(16,16,3, padding=0,bias=False) 
        self.bn5 = nn.BatchNorm2d(num_features=16)    
        self.drop = nn.Dropout2d(0)
        self.gpool = nn.AvgPool2d(5)

        self.conv6 = nn.Conv2d(16,10,1,padding=0,bias=False)

    def forward(self, x):
        x = self.drop(self.bn1(F.relu(self.conv1(x))))
        x = self.drop(self.bn2(F.relu(self.conv2(x))))
        x = self.pool1(self.tns1(x))

      

        x = self.drop(self.bn3(F.relu(self.conv3(x))))       
        x = self.drop(self.bn4(F.relu(self.conv4(x))))


        x = self.drop(self.bn5(F.relu(self.conv5(x))))

        x = self.gpool(x)
        x = self.conv6(x)
        x = x.view(-1, 10)
        return F.log_softmax(x,dim=-1)        