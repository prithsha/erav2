import torch
from torch import nn
import torch.nn.functional as F

# With regularization . Adding dropout 
# Adding global average pooling and capacity at bottom layer
class BatchNormalNeuralNetwork(nn.Module):

    def __init__(self, drop_out = 0.1):
        super().__init__()
        self.drop_out = drop_out
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1, bias=False),            
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )#

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(self.drop_out)
        )#
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=8,kernel_size=1,padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )#

        # Max pool
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )#16

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )#
        
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(self.drop_out)
        )#

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(self.drop_out)
        )#

        self.conv_block7 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=16,kernel_size=1,padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )#


        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )#8

        self.conv_block8 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )#8

        self.conv_block9 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )#8


        self.conv_block10 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(self.drop_out)
        )#4

        # Output block

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )


        self.conv_block11 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=10,kernel_size=1,padding=0, bias=False)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x) 
        
        x = self.pool1(x)
              
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.conv_block7(x)

        x = self.pool2(x)

        x = self.conv_block8(x)
        x = self.conv_block9(x)
        x = self.conv_block10(x)
        
        x = self.gap(x)
        x = self.conv_block11(x)
        
        x = x.view(-1,10)
       
        return F.log_softmax(x, dim=1)
