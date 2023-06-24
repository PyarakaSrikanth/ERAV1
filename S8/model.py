import torch.nn as nn
import torch.nn.functional as F

#C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self,norm='BN',groupsize=2,drop=0.01):
      super(Net,self).__init__()
      self.conv1 = nn.Conv2d(3, 10, 3, padding=1,bias=False)
      self.norm1 = self.select_norm(norm,10,groupsize)
      self.conv2 = nn.Conv2d(10, 10, 3, padding=1,bias=False)
      self.norm2 = self.select_norm(norm,10,groupsize)
      self.tns1 = nn.Conv2d(10,16,1,padding=0,bias=False)
      self.pool1 = nn.MaxPool2d(2, 2) 

      self.conv3 = nn.Conv2d(16, 24, 3, padding=1,bias=False)
      self.norm3 = self.select_norm(norm,24,groupsize)
      self.conv4 = nn.Conv2d(24, 16, 3, padding=1,bias=False)
      self.norm4 = self.select_norm(norm,16,groupsize)
      self.conv5 = nn.Conv2d(16, 32, 3, padding=1,bias=False)
      self.norm5 = self.select_norm(norm,32,groupsize)
      self.tns2 = nn.Conv2d(32,16,1,padding=0,bias=False)
      self.pool2 = nn.MaxPool2d(2, 2)

      self.conv6 = nn.Conv2d(16, 32, 3, padding=1,bias=False)
      self.norm6 = self.select_norm(norm,32,groupsize)
      self.conv7 = nn.Conv2d(32, 32, 3, padding=1,bias=False)
      self.norm7 = self.select_norm(norm,32,groupsize)
      self.conv8 = nn.Conv2d(32, 32, 3, padding=1,bias=False)
      self.norm8 = self.select_norm(norm,32,groupsize)

          
      self.drop = nn.Dropout2d(drop)
      self.gpool = nn.AvgPool2d(8)

      self.tns3 = nn.Conv2d(32,10,1,padding=0,bias=False)

    def forward(self, x):
        x = self.drop(self.norm1(F.relu(self.conv1(x))))
        x = self.drop(self.norm2(F.relu(self.conv2(x))))
        x = self.pool1(self.tns1(x))

        x = self.drop(self.norm3(F.relu(self.conv3(x))))
        x = self.drop(self.norm4(F.relu(self.conv4(x))))
        x = self.drop(self.norm5(F.relu(self.conv5(x))))
        x = self.pool2(self.tns2(x))

        x = self.drop(self.norm6(F.relu(self.conv6(x))))
        x = self.drop(self.norm7(F.relu(self.conv7(x))))
        x = self.drop(self.norm8(F.relu(self.conv8(x))))

        x = self.gpool(x)
        x = self.tns3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x,dim=-1) 
  
        
    def select_norm(self, norm, channels,groupsize=2):
        if norm == 'BN':
            return nn.BatchNorm2d(channels)
        elif norm == 'LN':
            return nn.GroupNorm(1,channels)
        elif norm == 'GN':
            return nn.GroupNorm(groupsize,channels)            