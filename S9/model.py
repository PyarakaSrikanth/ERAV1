import torch.nn as nn
import torch.nn.functional as F

#C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self,norm='BN',drop=0.01):
      super(Net,self).__init__()

      ##Block 1
      self.conv1 = nn.Conv2d(3, 16, 3, padding=1,bias=False)
      self.norm1 = self.select_norm(norm,16)
      self.conv2 = nn.Conv2d(16, 32, 3, padding=1,bias=False)
      self.norm2 = self.select_norm(norm,32)
      self.conv3 = nn.Conv2d(32, 32, 3, padding=1,bias=False)
      self.norm3 = self.select_norm(norm,32)
      self.dilated_conv1 = nn.Conv2d(32,32,3,stride = 2,dilation=2,padding=0,bias=False)
      self.dilated_norm1 = self.select_norm(norm,32)
      ##Block 2

      self.conv4 = nn.Conv2d(32, 32, 3, padding=1,bias=False)
      self.norm4 = self.select_norm(norm,32)
      self.conv5 = nn.Conv2d(32, 52, 3, padding=1,bias=False)
      self.norm5 = self.select_norm(norm,52)
      self.dilated_conv2 = nn.Conv2d(52,64,3,stride = 2,dilation=2,padding=0,bias=False)
      self.dilated_norm2 = self.select_norm(norm,64)
      ## Block 3
      #nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, groups=16)
      self.depthwise_conv1 = nn.Conv2d(64, 64,1,stride = 1,groups = 64, padding=0,bias=False)
      self.depthwise_norm1 = self.select_norm(norm,64)
      self.conv6 = nn.Conv2d(64, 64, 3, padding=1,bias=False)
      self.norm6 = self.select_norm(norm,64)
      self.strided_conv1 = nn.Conv2d(64,64,1,stride = 2,padding=1,bias=False)
      self.strided_norm1 = self.select_norm(norm,64)
      ## Block 4
      self.conv7 = nn.Conv2d(64, 64, 3, padding=1,bias=False)
      self.norm7 = self.select_norm(norm,64)
      self.conv8 = nn.Conv2d(64,10,3,stride = 1, padding=1,bias=False)

          
      self.drop = nn.Dropout2d(drop)

      self.gap = nn.AvgPool2d(4)

      

    def forward(self, x):
        x = self.drop(self.norm1(F.relu(self.conv1(x))))
        x = self.drop(self.norm2(F.relu(self.conv2(x))))
        x = self.drop(self.norm3(F.relu(self.conv3(x))))
        x = self.drop(self.dilated_norm1(F.relu(self.dilated_conv1(x))))

        x = self.drop(self.norm4(F.relu(self.conv4(x))))
        x = self.drop(self.norm5(F.relu(self.conv5(x))))
        x = self.drop(self.dilated_norm2(F.relu(self.dilated_conv2(x))))

   

        x = self.drop(self.depthwise_norm1(F.relu(self.depthwise_conv1(x))))
        x = self.drop(self.norm6(F.relu(self.conv6(x))))
        x = self.drop(self.strided_norm1(F.relu(self.strided_conv1(x))))
         
        x = self.drop(self.norm7(F.relu(self.conv7(x))))
        x = self.conv8(x)
        x = self.gap(x)
      
        x = x.view(-1, 10)
        return F.log_softmax(x,dim=-1) 
  
        
    def select_norm(self, norm, channels,groupsize=2):
        if norm == 'BN':
            return nn.BatchNorm2d(channels)
        elif norm == 'LN':
            return nn.GroupNorm(1,channels)
        elif norm == 'GN':
            return nn.GroupNorm(groupsize,channels)            