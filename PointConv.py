import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pointconv_util_groupnorm import PointConvDensitySetAbstraction

class PointConv(nn.Module):
    def __init__(self, normal_channel=False):
        super(PointConv, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        # self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        # self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[512], bandwidth = 0.4, group_all=True)
        
        
        self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[32], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=16, in_channel= 32 + 3, mlp=[64], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel= 64 + 3, mlp=[128], bandwidth = 0.4, group_all=True) # version 3-5

        # self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel= 64 + 3, mlp=[16], bandwidth = 0.2, group_all=True) # version feb
        
        
        #self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[128], bandwidth = 0.4, group_all=True)  

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.GroupNorm(1, 128)
        self.drop1 = nn.Dropout(0.5)



        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.GroupNorm(1, 64)
        self.drop3 = nn.Dropout(0.5)    

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.GroupNorm(1, 32)
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(32, 3)


    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l3_points.view(B, 128)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))



        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        # x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        # x = self.fc5(x)

        return x